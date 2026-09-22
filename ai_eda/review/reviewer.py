"""The 14 review areas, each a check that reads only the IR, the files on disk and the tool reports.

How evidence is read (see ``docs/ARCHITECTURE.md`` section 4):

* ``review.ir_vs_*``: the artifact must be generated from the current IR
  (``is_stale``) and unchanged on disk (``matches_disk``); ``ir_vs_pcb``
  additionally reports NOT_VERIFIED while any layout item's provenance still
  needs verification (unrecorded / assumption / llm_generated).
* ``review.pcb_vs_bom`` / ``review.pcb_vs_cpl``: the CSV must be fresh and
  unchanged on disk, the board artifact must be fresh and unchanged, and the
  rows are compared with the footprints read from the compiled
  ``.kicad_pcb`` (reference, value, footprint id for the BOM; reference,
  position, rotation, side for the CPL). Without a board there is nothing
  to compare with: NOT_VERIFIED, never PASS.
* ``review.erc`` / ``review.drc``: the latest tool result must have run on
  the current artifact; its status is passed through (any KiCad violation,
  warning included, is FAIL - ``ai_eda.tools.kicad.cli``).
* ``review.spice_vs_requirements``: NOT_VERIFIED without a simulation setup,
  netlist or result; FAIL ``regenerate`` (SPICE_NETLIST) when the netlist is
  stale or changed on disk; FAIL ``rerun_tool`` (``spice``) when the results
  artifact is stale, changed, or was produced for a different netlist than
  the current one (``results.json`` and the ``spice`` result both name the
  netlist hash); FAIL ``human`` when an expectation failed, has no result
  from this run, names a requirement that does not exist, or **claims to
  verify a requirement whose numeric value it does not agree with** (a 6 V
  requirement is not verified by an expectation whose nominal is 4 V: the
  nominal must match the requirement's value, same unit, within the
  expectation's tolerance); NOT_VERIFIED when an expectation is traced to a
  requirement without a comparable numeric value ("traced but not
  compared"), or when the netlist rests on an assumption; PASS only when
  every expectation of the current run passed and agrees with its
  requirement, with the rawfiles and ``results.json`` as evidence
  (expectations without a ``requirement_id`` are listed under
  ``details["untraced"]``). Every verdict says it holds at nominal component
  values and one temperature only (``details["conditions"]``).
* ``review.calculations_vs_design``: the reviewer recomputes every derived
  value itself (:func:`ai_eda.tools.calc.recompute_parameters`, the same
  deterministic calculators) and passes that verdict through - FAIL
  ``human`` with the mismatches, NOT_VERIFIED with what could not be
  recomputed - after the provenance-shape checks (a derived value must name
  a tool and inputs that exist). The CALCULATION stage's stored
  ``calc.recompute`` is reported next to it and a disagreement between the
  two is a FAIL of its own.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from ai_eda.ir import (
    ArtifactKind,
    ArtifactRef,
    CircuitIR,
    Evidence,
    ProvenanceKind,
    ValidationResult,
    ValidationStatus,
    worst_status,
)
from ai_eda.review.areas import ReviewArea
from ai_eda.tools.calc.recompute import CHECK_ID as CALC_CHECK_ID, recompute_parameters
from ai_eda.tools.kicad.board import BoardFootprint, read_board_footprints
from ai_eda.tools.kicad.geometry import normalize_angle
from ai_eda.tools.manufacturing.outputs import OUTPUT_CHECKS
from ai_eda.tools.spice.stage import CHECK_ID as SPICE_CHECK_ID, read_results

Check = Callable[[CircuitIR, Path], ValidationResult]

#: position tolerance when comparing CPL rows (4 decimals) with board coordinates (6 decimals)
POSITION_TOL_MM = 1e-4


class ReviewReport(BaseModel):
    ir_hash: str
    results: list[ValidationResult] = Field(default_factory=list)

    @property
    def overall(self) -> ValidationStatus:
        return worst_status(r.status for r in self.results)

    @property
    def failures(self) -> list[ValidationResult]:
        return [r for r in self.results if r.status == ValidationStatus.FAIL]

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return f"review {self.overall}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


class IndependentReviewer:
    """Runs one check per :class:`ReviewArea`.

    Checks that need a tool report or an artifact that does not exist return
    NOT_VERIFIED. Checks that find a mismatch return FAIL with details naming
    both sides, so the repair loop can pick a deterministic strategy.
    """

    tool_id = "independent_reviewer"
    version = "0.1"

    def __init__(self, tools: dict[str, Any] | None = None) -> None:
        self.tools = tools or {}
        self._checks: dict[ReviewArea, Check] = {
            ReviewArea.REQUIREMENTS_VS_IR: self.check_requirements_vs_ir,
            ReviewArea.IR_VS_SCHEMATIC: self._artifact_freshness(ArtifactKind.SCHEMATIC),
            ReviewArea.IR_VS_PCB: self.check_ir_vs_pcb,
            ReviewArea.SCHEMATIC_VS_PCB: self.check_schematic_vs_pcb,
            ReviewArea.PCB_VS_BOM: self.check_pcb_vs_bom,
            ReviewArea.PCB_VS_CPL: self.check_pcb_vs_cpl,
            ReviewArea.MANUFACTURING_OUTPUTS: self.check_manufacturing_outputs,
            ReviewArea.CALCULATIONS_VS_DESIGN: self.check_calculations_vs_design,
            ReviewArea.SPICE_VS_REQUIREMENTS: self.check_spice_vs_requirements,
            ReviewArea.ERC: self._tool_result_fresh("kicad.erc", ArtifactKind.SCHEMATIC),
            ReviewArea.DRC: self._tool_result_fresh("kicad.drc", ArtifactKind.PCB),
            ReviewArea.REGULATORY_PROVENANCE: self.check_regulatory_provenance,
            ReviewArea.COMPONENT_PROVENANCE: self.check_component_provenance,
            ReviewArea.MANUFACTURING_CAPABILITIES: self._latest_status("mfg.capability"),
        }

    def review(self, ir: CircuitIR, workdir: Path) -> ReviewReport:
        ir_hash = ir.content_hash()
        report = ReviewReport(ir_hash=ir_hash)
        for area, check in self._checks.items():
            try:
                r = check(ir, workdir)
            except Exception as e:  # a crashing check is UNRESOLVED, never silently PASS
                r = ValidationResult(check_id=area, status=ValidationStatus.UNRESOLVED, message=f"check raised: {e!r}")
            r.check_id = area
            r.tool = r.tool or self.tool_id
            r.tool_version = r.tool_version or self.version
            r.ir_hash = ir_hash
            report.results.append(r)
        return report

    # --- generic check factories ------------------------------------------------

    @staticmethod
    def _stale_or_changed(ir: CircuitIR, kind: ArtifactKind, art: ArtifactRef) -> ValidationResult | None:
        """FAIL (repair: regenerate ``kind``) when ``art`` is not evidence about the current IR, else None."""
        if art.is_stale(ir.content_hash()):
            return ValidationResult(check_id="", status=ValidationStatus.FAIL, message=f"{kind} was generated from a different IR version", details={"artifact": kind, "repair": "regenerate"})
        if not art.matches_disk():
            return ValidationResult(check_id="", status=ValidationStatus.FAIL, message=f"{kind} on disk does not match recorded hash", details={"artifact": kind, "repair": "regenerate"})
        return None

    @staticmethod
    def _evidence(what: str, art: ArtifactRef) -> Evidence:
        return Evidence(description=what, path=art.path, content_hash=art.disk_hash())

    def _artifact_freshness(self, kind: ArtifactKind) -> Check:
        def check(ir: CircuitIR, workdir: Path) -> ValidationResult:
            art = ir.artifacts.get(kind)
            if art is None:
                return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message=f"no {kind} artifact")
            problem = self._stale_or_changed(ir, kind, art)
            if problem is not None:
                return problem
            return ValidationResult(check_id="", status=ValidationStatus.PASS, message=f"{kind} matches IR {ir.content_hash()[:16]}", evidence=[self._evidence(str(kind), art)])
        return check

    def check_ir_vs_pcb(self, ir: CircuitIR, workdir: Path) -> ValidationResult:
        """Board freshness, then layout traceability: every placement / track / via / zone must say where it came from."""
        res = self._artifact_freshness(ArtifactKind.PCB)(ir, workdir)
        if res.status is not ValidationStatus.PASS or ir.pcb is None:
            return res
        unverified = [label for label, prov in ir.pcb.layout_items() if prov.needs_verification]
        if unverified:
            return ValidationResult(
                check_id="",
                status=ValidationStatus.NOT_VERIFIED,
                message=f"{len(unverified)} layout item(s) have unrecorded / assumed / model-generated origin",
                details={"unverified_layout": unverified, "repair": "human"},
                evidence=res.evidence,
            )
        return res

    def _tool_result_fresh(self, check_id: str, on_kind: ArtifactKind) -> Check:
        def check(ir: CircuitIR, workdir: Path) -> ValidationResult:
            res = ir.validation.latest(check_id)
            art = ir.artifacts.get(on_kind)
            if res is None or art is None:
                return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message=f"{check_id} has not been run")
            if res.artifact_hash != art.content_hash:
                return ValidationResult(check_id="", status=ValidationStatus.FAIL, message=f"{check_id} report is stale (ran on a different {on_kind})", details={"tool_check": check_id, "repair": "rerun_tool"})
            details: dict = {}
            if res.status is ValidationStatus.FAIL:
                # a violation in the design (error or warning) is not something regeneration fixes
                details = {
                    "repair": "human",
                    "error_types": sorted({str(v.get("type")) for v in res.details.get("errors", [])}),
                    "warning_types": sorted({str(v.get("type")) for v in res.details.get("warnings", [])}),
                }
            return ValidationResult(check_id="", status=res.status, message=f"{check_id}: {res.message}", evidence=res.evidence, details=details)
        return check

    def _tool_result_present(self, check_id: str, kind: ArtifactKind) -> Check:
        def check(ir: CircuitIR, workdir: Path) -> ValidationResult:
            res = ir.validation.latest(check_id)
            if res is None:
                return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message=f"no {check_id} result")
            return ValidationResult(check_id="", status=res.status, message=res.message)
        return check

    def _latest_status(self, check_id: str) -> Check:
        return self._tool_result_present(check_id, ArtifactKind.REVIEW_REPORT)

    # --- specific checks ----------------------------------------------------------

    def check_requirements_vs_ir(self, ir: CircuitIR, workdir: Path) -> ValidationResult:
        """Every *authoritative* design requirement is served by a component or net.

        A requirement whose value still needs verification (``llm_generated``
        - a model's implicit inference the user has not accepted - or an
        ``assumption``) is neither enforced nor satisfiable here: it does not
        FAIL the design when unserved, and a component that lists it in
        ``serves_requirements`` does not make anything PASS. While any such
        requirement exists the verdict is NOT_VERIFIED, naming them.
        """
        reqs = ir.requirements
        if not reqs.can_proceed:
            return ValidationResult(check_id="", status=ValidationStatus.USER_INPUT_REQUIRED, message=f"{len(reqs.blocking_questions)} open question(s), {len(reqs.conflicts)} conflict(s)")
        if not reqs.requirements:
            return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message="no structured requirements")
        served = {rid for c in ir.components for rid in c.serves_requirements}
        served |= {rid for n in ir.nets for rid in n.serves_requirements}
        # Only design-level requirements must map to components / nets; application / regulatory
        # requirements are covered by the regulatory and manufacturing checks.
        design_categories = {"electrical", "thermal", "mechanical", "signal_integrity", "power_integrity", "rf"}
        unverified = [r.id for r in reqs.requirements if r.value is not None and r.value.provenance.needs_verification]
        authoritative = [r for r in reqs.requirements if r.id not in set(unverified)]
        unserved = [r.id for r in authoritative if r.category in design_categories and r.id not in served]
        if unserved:
            return ValidationResult(
                check_id="", status=ValidationStatus.FAIL, message="requirements not traced to any component",
                details={"unserved": unserved, "unverified": unverified, "repair": "human"},
            )
        if unverified:
            return ValidationResult(
                check_id="", status=ValidationStatus.NOT_VERIFIED,
                message=f"{len(unverified)} model-inferred / assumed requirement(s) not yet accepted by the user; they are neither enforced nor counted as served",
                details={"unverified": unverified, "served_but_unverified": sorted(set(unverified) & served)},
            )
        return ValidationResult(check_id="", status=ValidationStatus.PASS)

    def check_schematic_vs_pcb(self, ir: CircuitIR, workdir: Path) -> ValidationResult:
        """Schematic <-> PCB consistency from the real ``kicad-cli pcb drc --schematic-parity`` evidence.

        PASS only when the latest DRC (a) ran on the current board artifact,
        (b) actually evaluated parity, (c) against the current schematic
        artifact, and (d) found no parity issue. A parity finding of any KiCad
        severity is a FAIL here: two artifacts compiled from one IR must agree.
        """
        sch = ir.artifacts.get(ArtifactKind.SCHEMATIC)
        pcb = ir.artifacts.get(ArtifactKind.PCB)
        if sch is None or pcb is None:
            return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message="schematic or pcb artifact missing")
        drc = ir.validation.latest("kicad.drc")
        if drc is None:
            return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message="kicad.drc has not been run")
        if drc.artifact_hash != pcb.content_hash:
            return ValidationResult(check_id="", status=ValidationStatus.FAIL, message="kicad.drc ran on a different board than the current PCB artifact", details={"tool_check": "kicad.drc", "repair": "rerun_tool"})
        if not drc.details.get("schematic_parity_checked"):
            return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message=f"DRC schematic parity not evaluated ({drc.details.get('schematic_parity_reason', 'no reason recorded')})")
        if drc.details.get("schematic_hash") != sch.content_hash:
            return ValidationResult(check_id="", status=ValidationStatus.FAIL, message="schematic parity was evaluated against a different schematic than the current artifact", details={"tool_check": "kicad.drc", "repair": "rerun_tool"})
        parity = list(drc.details.get("schematic_parity", []))
        if parity:
            types = sorted({str(v.get("type")) for v in parity})
            return ValidationResult(
                check_id="",
                status=ValidationStatus.FAIL,
                message=f"{len(parity)} schematic/PCB parity issue(s): {types}",
                details={"parity_violations": parity, "repair": "human"},
                evidence=drc.evidence,
            )
        return ValidationResult(check_id="", status=ValidationStatus.PASS, message="kicad.drc schematic parity: 0 issues against the current schematic and board", evidence=drc.evidence)

    def check_manufacturing_outputs(self, ir: CircuitIR, workdir: Path) -> ValidationResult:
        """Gerber + drill artifacts must be fresh, on disk, and format-checked on exactly these files."""
        stale: list[ArtifactKind] = []
        missing: list[ArtifactKind] = []
        for kind in OUTPUT_CHECKS:
            art = ir.artifacts.get(kind)
            if art is None:
                missing.append(kind)
            elif art.is_stale(ir.content_hash()) or not art.matches_disk():
                stale.append(kind)
        if stale:
            return ValidationResult(
                check_id="",
                status=ValidationStatus.FAIL,
                message=f"{[str(k) for k in stale]} generated from a different IR version or changed on disk",
                details={"artifact": stale[0], "artifacts": list(stale), "repair": "regenerate"},
            )
        if missing:
            return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message=f"no {[str(k) for k in missing]} artifact")
        unchecked: list[str] = []
        stale_checks: list[str] = []
        failed: list[str] = []
        for kind, check_id in OUTPUT_CHECKS.items():
            res = ir.validation.latest(check_id)
            if res is None or not res.is_tool_backed:
                unchecked.append(check_id)
            elif res.artifact_hash != ir.artifacts[kind].content_hash:
                stale_checks.append(check_id)
            elif res.status != ValidationStatus.PASS:
                failed.append(f"{check_id}: {res.message}")
        if stale_checks:
            return ValidationResult(
                check_id="",
                status=ValidationStatus.FAIL,
                message=f"{stale_checks} ran on different files than the current artifacts",
                details={"tool_check": stale_checks[0], "tool_checks": list(stale_checks), "repair": "rerun_tool"},
            )
        if unchecked:
            return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message=f"{unchecked} not run")
        if failed:
            return ValidationResult(check_id="", status=ValidationStatus.FAIL, message="; ".join(failed))
        return ValidationResult(check_id="", status=ValidationStatus.PASS, message=f"gerber + drill fresh and checked for IR {ir.content_hash()[:16]}")

    @staticmethod
    def _csv_rows(path: str) -> list[dict[str, str]]:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _csv_artifact(
        self, ir: CircuitIR, kind: ArtifactKind, key: str, expected_refs: set[str], only_in_csv: str, only_in: str, what: str
    ) -> tuple[ArtifactRef, list[dict[str, str]]] | ValidationResult:
        """The fresh, unchanged CSV artifact of ``kind`` and its rows, or the review result that says why not."""
        art = ir.artifacts.get(kind)
        if art is None or not Path(art.path).exists():
            return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message=f"no {kind} artifact")
        rows = self._csv_rows(art.path)
        refs = {row[key] for row in rows}
        regenerate = {"artifact": kind, "repair": "regenerate"}
        evidence = [self._evidence(f"{kind} csv", art)]
        if refs != expected_refs:
            return ValidationResult(
                check_id="",
                status=ValidationStatus.FAIL,
                message=f"{kind} {what} differ from the IR",
                details={only_in_csv: sorted(refs - expected_refs), only_in: sorted(expected_refs - refs), **regenerate},
                evidence=evidence,
            )
        if art.is_stale(ir.content_hash()):
            return ValidationResult(check_id="", status=ValidationStatus.FAIL, message=f"{kind} generated from an older IR", details=regenerate, evidence=evidence)
        if not art.matches_disk():
            return ValidationResult(check_id="", status=ValidationStatus.FAIL, message=f"{kind} on disk does not match its recorded hash (edited after generation?)", details=regenerate, evidence=evidence)
        return art, rows

    def _fresh_board(self, ir: CircuitIR) -> tuple[ArtifactRef, list[BoardFootprint]] | ValidationResult:
        """The fresh, unchanged board artifact and its footprints, or the result that says why there is none."""
        pcb = ir.artifacts.get(ArtifactKind.PCB)
        if pcb is None or not Path(pcb.path).exists():
            return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message="no PCB artifact to compare with")
        problem = self._stale_or_changed(ir, ArtifactKind.PCB, pcb)
        if problem is not None:
            problem.message = "cannot compare with the board: " + problem.message
            return problem
        return pcb, read_board_footprints(Path(pcb.path))

    def check_pcb_vs_bom(self, ir: CircuitIR, workdir: Path) -> ValidationResult:
        """BOM rows vs the footprints of the compiled board: reference, value and footprint id must agree."""
        got = self._csv_artifact(ir, ArtifactKind.BOM, "Reference", {c.ref for c in ir.components}, "only_in_bom", "only_in_ir", "references")
        if isinstance(got, ValidationResult):
            return got
        bom, rows = got
        board = self._fresh_board(ir)
        if isinstance(board, ValidationResult):
            board.message = f"BOM matches the IR ({len(rows)} rows) but is unverified against a board: " + board.message
            board.evidence = [self._evidence("bom csv", bom)]
            return board
        pcb, footprints = board
        on_board = {fp.ref: fp for fp in footprints}
        mismatches: list[str] = []
        for row in rows:
            fp = on_board.get(row["Reference"])
            if fp is None:
                mismatches.append(f"{row['Reference']}: in BOM, not on the board")
                continue
            if row["Value"] != fp.value:
                mismatches.append(f"{fp.ref}: value BOM {row['Value']!r} vs board {fp.value!r}")
            if row["Footprint"] != fp.lib_id:
                mismatches.append(f"{fp.ref}: footprint BOM {row['Footprint']!r} vs board {fp.lib_id!r}")
        for ref in sorted(set(on_board) - {row["Reference"] for row in rows}):
            mismatches.append(f"{ref}: on the board, not in BOM")
        evidence = [self._evidence("bom csv", bom), self._evidence("kicad_pcb", pcb)]
        if mismatches:
            return ValidationResult(check_id="", status=ValidationStatus.FAIL, message="BOM and board disagree", details={"mismatches": mismatches, "repair": "human"}, evidence=evidence)
        return ValidationResult(check_id="", status=ValidationStatus.PASS, message=f"{len(rows)} BOM rows match the board's footprints (reference, value, footprint)", evidence=evidence)

    def check_pcb_vs_cpl(self, ir: CircuitIR, workdir: Path) -> ValidationResult:
        """CPL rows vs the footprints of the compiled board: designator, position, rotation and side must agree."""
        if ir.pcb is None:
            return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message="no PCB design in the IR")
        got = self._csv_artifact(ir, ArtifactKind.CPL, "Designator", {p.component_ref for p in ir.pcb.placements}, "only_in_cpl", "only_in_pcb", "designators")
        if isinstance(got, ValidationResult):
            return got
        cpl, rows = got
        board = self._fresh_board(ir)
        if isinstance(board, ValidationResult):
            board.message = f"CPL matches the IR placements ({len(rows)} rows) but is unverified against a board: " + board.message
            board.evidence = [self._evidence("cpl csv", cpl)]
            return board
        pcb, footprints = board
        on_board = {fp.ref: fp for fp in footprints}
        mismatches: list[str] = []
        for row in rows:
            fp = on_board.get(row["Designator"])
            if fp is None:
                mismatches.append(f"{row['Designator']}: in CPL, not on the board")
                continue
            try:
                x, y = float(row["Mid X"].rstrip("m")), float(row["Mid Y"].rstrip("m"))
                rot = float(row["Rotation"])
            except ValueError as exc:
                mismatches.append(f"{fp.ref}: unreadable CPL row {row!r} ({exc})")
                continue
            if not (math.isclose(x, fp.x, abs_tol=POSITION_TOL_MM) and math.isclose(y, fp.y, abs_tol=POSITION_TOL_MM)):
                mismatches.append(f"{fp.ref}: position CPL ({x}, {y}) vs board ({fp.x}, {fp.y})")
            if not math.isclose(normalize_angle(rot), normalize_angle(fp.rotation), abs_tol=1e-6):
                mismatches.append(f"{fp.ref}: rotation CPL {rot} vs board {fp.rotation}")
            if row["Layer"] != fp.side:
                mismatches.append(f"{fp.ref}: side CPL {row['Layer']!r} vs board {fp.side!r}")
        for ref in sorted(set(on_board) - {row["Designator"] for row in rows}):
            mismatches.append(f"{ref}: on the board, not in CPL")
        evidence = [self._evidence("cpl csv", cpl), self._evidence("kicad_pcb", pcb)]
        if mismatches:
            return ValidationResult(check_id="", status=ValidationStatus.FAIL, message="CPL and board disagree", details={"mismatches": mismatches, "repair": "human"}, evidence=evidence)
        return ValidationResult(check_id="", status=ValidationStatus.PASS, message=f"{len(rows)} CPL rows match the board's footprints (position, rotation, side)", evidence=evidence)

    def check_spice_vs_requirements(self, ir: CircuitIR, workdir: Path) -> ValidationResult:
        """Every expectation of the simulation setup must have passed in a run of the *current* netlist (see the module docstring)."""
        setup = ir.simulation
        if setup is None:
            return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message="no simulation setup in the IR")
        netlist = ir.artifacts.get(ArtifactKind.SPICE_NETLIST)
        if netlist is None:
            return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message="no SPICE netlist artifact (the SPICE stage has not compiled one)")
        problem = self._stale_or_changed(ir, ArtifactKind.SPICE_NETLIST, netlist)
        if problem is not None:
            return problem
        spice = ir.validation.latest(SPICE_CHECK_ID)
        results = ir.artifacts.get(ArtifactKind.SPICE_RESULT)
        if spice is None or not spice.is_tool_backed or results is None:
            return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message="spice has not been run on this netlist")
        rerun = {"tool_check": SPICE_CHECK_ID, "repair": "rerun_tool"}
        evidence = [self._evidence("spice_netlist", netlist), self._evidence("spice results.json", results)]
        if results.is_stale(ir.content_hash()):
            return ValidationResult(check_id="", status=ValidationStatus.FAIL, message="spice results were produced for a different IR version", details=rerun, evidence=evidence)
        if not results.matches_disk():
            return ValidationResult(check_id="", status=ValidationStatus.FAIL, message="spice results.json on disk does not match its recorded hash", details=rerun, evidence=evidence)
        if spice.artifact_hash != netlist.content_hash:
            return ValidationResult(check_id="", status=ValidationStatus.FAIL, message="the latest spice result ran on a different netlist than the current artifact", details=rerun, evidence=evidence)
        try:
            data = read_results(results.path)
        except ValueError as e:
            return ValidationResult(check_id="", status=ValidationStatus.FAIL, message=f"spice results.json is not in the current layout ({e}); re-run", details=rerun, evidence=evidence)
        if data.get("netlist_hash") != netlist.content_hash:
            return ValidationResult(check_id="", status=ValidationStatus.FAIL, message="results.json names a different netlist hash than the current artifact", details=rerun, evidence=evidence)
        evidence += [e for e in spice.evidence if e.path not in {x.path for x in evidence}]
        requirements = {r.id: r for r in ir.requirements.requirements}
        failed: list[str] = []
        missing: list[str] = []
        not_passed: list[tuple[ValidationStatus, str]] = []
        unknown_req: list[str] = []
        untraced: list[str] = []
        not_compared: list[str] = []
        disagree: list[str] = []
        verified: dict[str, str | None] = {}
        for exp in setup.expectations:
            r = ir.validation.latest(f"{SPICE_CHECK_ID}.{exp.id}")
            if r is None or r.artifact_hash != netlist.content_hash:
                missing.append(exp.id)
            elif r.status is ValidationStatus.FAIL:
                failed.append(f"{exp.id}: {r.message}")
            elif r.status is not ValidationStatus.PASS:
                not_passed.append((r.status, f"{exp.id}: {r.status} ({r.message})"))
            else:
                verified[exp.id] = exp.requirement_id
            if exp.requirement_id is None:
                untraced.append(exp.id)
            elif exp.requirement_id not in requirements:
                unknown_req.append(f"{exp.id} -> {exp.requirement_id}")
            else:
                problem, comparable = self._nominal_vs_requirement(exp, requirements[exp.requirement_id])
                if problem is not None and comparable:
                    disagree.append(f"{exp.id}: {problem}")
                elif problem is not None:
                    not_compared.append(f"{exp.id}: {problem}")
        assumptions = list(spice.details.get("assumptions") or [])
        details: dict = {
            "verified": verified, "untraced": untraced, "not_compared": not_compared, "assumptions": assumptions,
            "conditions": spice.details.get("conditions"), "engine": data.get("engine"), "engine_version": data.get("engine_version"),
            "netlist_hash": netlist.content_hash,
        }
        if spice.status is ValidationStatus.FAIL and not failed:
            failed.append(f"spice: {spice.message}")
        if failed or missing or unknown_req or disagree:
            details.update({"repair": "human", "failed": failed, "no_result": missing, "unknown_requirements": unknown_req, "nominal_vs_requirement": disagree})
            parts = []
            if failed:
                parts.append(f"{len(failed)} expectation(s) failed")
            if missing:
                parts.append(f"no result from this run for {missing}")
            if unknown_req:
                parts.append(f"expectation(s) name unknown requirements {unknown_req}")
            if disagree:
                parts.append(f"{len(disagree)} expectation nominal(s) are not the requirement they claim to verify: {disagree}")
            return ValidationResult(check_id="", status=ValidationStatus.FAIL, message="; ".join(parts), details=details, evidence=evidence)
        if not_passed:
            details["not_passed"] = [m for _, m in not_passed]
            return ValidationResult(check_id="", status=worst_status(s for s, _ in not_passed), message="; ".join(m for _, m in not_passed), details=details, evidence=evidence)
        if not setup.expectations or spice.status is not ValidationStatus.PASS:
            return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message=f"spice ran but verified nothing: {spice.message}", details=details, evidence=evidence)
        if assumptions:
            return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message=f"the simulated netlist rests on unconfirmed assumption(s) {assumptions}: not evidence", details=details, evidence=evidence)
        if not_compared:
            return ValidationResult(
                check_id="", status=ValidationStatus.NOT_VERIFIED,
                message=f"{len(verified)} expectation(s) passed, but {len(not_compared)} are traced to requirements without a comparable value: {not_compared}",
                details=details, evidence=evidence,
            )
        traced = sum(1 for v in verified.values() if v is not None)
        return ValidationResult(
            check_id="",
            status=ValidationStatus.PASS,
            message=(
                f"{len(verified)} expectation(s) verified by {data.get('engine')} {data.get('engine_version')} on the current netlist "
                f"({traced} traced to requirements, nominals agree with the requirement values) - at nominal component values and one temperature only"
            ),
            details=details,
            evidence=evidence,
        )

    @staticmethod
    def _nominal_vs_requirement(exp, req) -> tuple[str | None, bool]:
        """``(problem, comparable)``: whether ``exp.nominal`` is the value ``req`` asks for.

        ``(None, True)`` when they agree within the expectation's tolerance;
        ``(why, True)`` when both are numbers in the same unit and disagree;
        ``(why, False)`` when the requirement has no numeric value or another
        unit, so nothing can be compared.
        """
        value = req.value
        if value is None:
            return f"requirement {req.id} has no value to compare the nominal with", False
        if isinstance(value.value, bool) or not isinstance(value.value, (int, float)):
            return f"requirement {req.id} value {value.value!r} is not a number", False
        unit_e, unit_r = (exp.nominal.unit or "").strip().lower(), (value.unit or "").strip().lower()
        if unit_e and unit_r and unit_e != unit_r:
            return f"nominal unit {exp.nominal.unit!r} is not the requirement's {value.unit!r}", False
        nominal, target = float(exp.nominal.value), float(value.value)
        limits = [abs(float(exp.tol_abs.value))] if exp.tol_abs is not None else []
        if exp.tol_rel is not None and target != 0.0:
            limits.append(abs(float(exp.tol_rel.value)) * abs(target))
        limit = max(limits) if limits else 1e-9 * max(1.0, abs(target))
        if abs(nominal - target) <= limit:
            return None, True
        unit = f" {value.unit}" if value.unit else ""
        return f"nominal {nominal:.6g}{unit} is not requirement {req.id}'s {target:.6g}{unit} (+/- {limit:.3g}{unit})", True

    def check_calculations_vs_design(self, ir: CircuitIR, workdir: Path) -> ValidationResult:
        """Every derived value must name a tool and inputs that exist, and the reviewer's own recompute must agree with it."""
        broken: list[str] = []
        for key, t in ir.parameters.items():
            p = t.provenance
            if p.kind == ProvenanceKind.DERIVED:
                if not p.tool:
                    broken.append(f"{key}: derived without tool")
                for src in p.derived_from:
                    if src not in ir.parameters and ir.requirements.get(src) is None:
                        broken.append(f"{key}: input '{src}' not found in IR")
        if broken:
            return ValidationResult(check_id="", status=ValidationStatus.FAIL, message="calculation provenance broken", details={"broken": broken, "repair": "human"})
        live = recompute_parameters(ir)  # the reviewer's own second opinion, from the same deterministic calculators
        stored = ir.validation.latest(CALC_CHECK_ID)
        details: dict = {
            "recompute": {"status": live.status.value, "mismatches": live.details.get("mismatches", []), "unverified": live.details.get("unverified", []), "parameters": live.details.get("parameters", {})},
            "stage_result": None if stored is None else {"status": stored.status.value, "ir_hash": stored.ir_hash, "current": stored.ir_hash == ir.content_hash(), "tool": stored.tool},
        }
        if live.status is ValidationStatus.FAIL:
            details["repair"] = "human"
            return ValidationResult(check_id="", status=ValidationStatus.FAIL, message=f"recompute: {live.message}", details=details)
        if stored is not None and stored.is_tool_backed and stored.ir_hash == ir.content_hash() and stored.status is not live.status:
            details["repair"] = "human"
            return ValidationResult(
                check_id="", status=ValidationStatus.FAIL,
                message=f"the stored {CALC_CHECK_ID} result says {stored.status.value} for this IR, the reviewer's recompute says {live.status.value}: {live.message}",
                details=details,
            )
        if live.status is not ValidationStatus.PASS:
            return ValidationResult(check_id="", status=live.status, message=f"recompute: {live.message}", details=details)
        return ValidationResult(check_id="", status=ValidationStatus.PASS, message=f"recompute: {live.message} (agrees with the stored values)", details=details)

    def check_regulatory_provenance(self, ir: CircuitIR, workdir: Path) -> ValidationResult:
        reg = ir.regulatory
        if not reg.jurisdiction_known:
            return ValidationResult(check_id="", status=ValidationStatus.USER_INPUT_REQUIRED, message="jurisdiction unknown")
        if not reg.requirements:
            return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message="no regulatory requirements researched")
        incomplete = [r.id for r in reg.requirements if not (r.provenance.source_url and r.provenance.retrieved_at and r.provenance.content_hash)]
        if incomplete:
            return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message="regulatory items lack url/date/hash provenance", details={"incomplete": incomplete})
        return ValidationResult(check_id="", status=worst_status(r.status for r in reg.requirements))

    def check_component_provenance(self, ir: CircuitIR, workdir: Path) -> ValidationResult:
        """Every component needs an authoritative identity (MPN backed by official data) and verified library entries."""
        weak: list[str] = []
        for c in ir.components:
            if not c.has_authoritative_identity:
                weak.append(f"{c.ref}.mpn[{c.mpn.provenance.kind if c.mpn is not None else 'missing'}]")
            if c.symbol and not c.symbol.verified:
                weak.append(f"{c.ref}.symbol[unverified]")
            if c.footprint and not c.footprint.verified:
                weak.append(f"{c.ref}.footprint[unverified]")
        if weak:
            return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message="component identity not fully verified", details={"weak": weak, "repair": "human"})
        if not ir.components:
            return ValidationResult(check_id="", status=ValidationStatus.NOT_VERIFIED, message="no components")
        return ValidationResult(check_id="", status=ValidationStatus.PASS)
