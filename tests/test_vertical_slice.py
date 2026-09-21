"""The vertical slice, end to end, against the real kicad-cli.

IR (divider + 3-pin header, naive tracks) -> Orchestrator.run:
  SCHEMATIC PASS -> ERC PASS -> PCB PASS -> DRC (+ schematic parity) PASS
  -> gerber + drill export + format checks PASS -> independent review
  -> RELEASE not PASS (regulatory / SPICE / fab capability are NOT_VERIFIED,
  which is the correct verdict for a design nobody simulated or researched).

Then the staleness scenario: the design changes after the run (a component
with its nets, placement and tracks is added), the reviewer flags every
derived artifact as stale, and the repair loop regenerates them and re-runs
ERC / DRC / output checks until the review is clean again - without ever
touching the IR.

Variant chosen for the staleness scenario: the appended component is wired
into the design (R3 = 1k from VOUT to GND, placed and routed so the naive
router stays DRC-clean), so the regenerated artifacts are *valid* and the
loop is expected to converge to PASS on the artifact / tool checks. This is
deterministic: every step is a pure function of the IR plus kicad-cli, and
the layout was validated with kicad-cli 10.0.6. (The alternative - leaving
the new component unconnected and asserting the loop reports the resulting
ERC/DRC failures as unresolved - would also be honest, but converging to
PASS proves more of the repair machinery.)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_eda.agents import AgentContext
from ai_eda.ir import (
    ArtifactKind,
    BoardSide,
    CircuitIR,
    Net,
    NetKind,
    PinRef,
    Placement,
    Provenance,
    ProvenanceKind,
    ValidationStatus,
)
from ai_eda.repair import RepairLoop
from ai_eda.review import IndependentReviewer, ReviewArea
from ai_eda.tools.kicad import KicadCli, KicadLibrary
from ai_eda.tools.routing import route_naive
from ai_eda.tools.spice import NgspiceRunner
from ai_eda.workflow import Orchestrator, PipelineState, Stage
from tests.conftest import make_component
from tests.fixtures_kicad import HAND_PLACED, PROJECT_ID, divider_with_connector_ir

LIB = KicadLibrary()
HAS_LIBS = LIB.footprint_file("Resistor_SMD", "R_0603_1608Metric") is not None and LIB.symbol_file("Device") is not None
kicad = KicadCli()
pytestmark = pytest.mark.skipif(not (kicad.available() and HAS_LIBS), reason="kicad-cli / KiCad libraries not installed")

S = ValidationStatus

#: review areas the vertical slice must prove with real evidence
PROVEN_AREAS = [
    ReviewArea.IR_VS_SCHEMATIC,
    ReviewArea.IR_VS_PCB,
    ReviewArea.SCHEMATIC_VS_PCB,
    ReviewArea.PCB_VS_BOM,
    ReviewArea.PCB_VS_CPL,
    ReviewArea.MANUFACTURING_OUTPUTS,
    ReviewArea.ERC,
    ReviewArea.DRC,
]


def _context(tmp_path: Path) -> AgentContext:
    return AgentContext(
        workdir=tmp_path,
        tools={"kicad_cli": kicad, "kicad_library": LIB, "spice": NgspiceRunner()},
        answers={"application": "bench voltage divider", "jurisdiction": "EU"},
    )


def _routed_ir(tmp_path: Path) -> CircuitIR:
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.pcb.tracks = route_naive(ir, LIB)  # the orchestrator compiles what the IR contains; it never routes
    return ir


def _run(tmp_path: Path) -> tuple[CircuitIR, PipelineState, AgentContext]:
    ir = _routed_ir(tmp_path)
    ctx = _context(tmp_path)
    state = Orchestrator(ctx).run(ir)
    for o in state.outcomes:
        print(f"{o.stage:<24} {o.status:<14} {o.message}")
    return ir, state, ctx


def _review(ir: CircuitIR, workdir: Path, tools: dict) -> dict[str, ValidationStatus]:
    report = IndependentReviewer(tools=tools).review(ir, workdir)
    for r in report.results:
        print(f"  {r.check_id:<34} {r.status:<14} {r.message}")
    return {r.check_id: r for r in report.results}


def add_r3_vout_to_gnd(ir: CircuitIR) -> None:
    """Append R3 (1k, VOUT -> GND) with placement and re-routed tracks - a valid design change."""
    ir.components.append(make_component("R3", "1k"))
    ir.net("VOUT").pins.append(PinRef(component_ref="R3", pin_number="1"))
    ir.net("GND").pins.append(PinRef(component_ref="R3", pin_number="2"))
    ir.pcb.placements.append(Placement(component_ref="R3", x_mm=20.0, y_mm=11.08, rotation_deg=270.0, side=BoardSide.TOP, provenance=HAND_PLACED))
    ir.pcb.tracks = route_naive(ir, LIB)


# --------------------------------------------------------------------------- the run


def test_pipeline_runs_the_slice_with_real_tools(tmp_path: Path):
    ir, state, ctx = _run(tmp_path)
    assert not state.blocked
    outcomes = {o.stage: o for o in state.outcomes}
    assert [o.stage for o in state.outcomes] == list(Stage)

    for stage in (Stage.SCHEMATIC, Stage.ERC, Stage.PCB, Stage.DRC, Stage.MANUFACTURING_OUTPUTS):
        assert outcomes[stage].status is S.PASS, f"{stage}: {outcomes[stage].message}"
    assert outcomes[Stage.DRC].message.startswith("0 error(s), 0 warning(s); schematic parity: 0 issue(s)")
    assert "mfg.gerber PASS" in outcomes[Stage.MANUFACTURING_OUTPUTS].message
    assert "mfg.drill PASS" in outcomes[Stage.MANUFACTURING_OUTPUTS].message

    # artifacts: all six, all fresh, all on disk, schematic and board are siblings with the project stem
    kinds = {ArtifactKind.SCHEMATIC, ArtifactKind.PCB, ArtifactKind.BOM, ArtifactKind.CPL, ArtifactKind.GERBER, ArtifactKind.DRILL}
    assert kinds <= set(ir.artifacts)
    ir_hash = ir.content_hash()
    for kind in kinds:
        art = ir.artifacts[kind]
        assert art.generated_from_ir_hash == ir_hash and art.matches_disk(), kind
    sch, pcb = Path(ir.artifacts[ArtifactKind.SCHEMATIC].path), Path(ir.artifacts[ArtifactKind.PCB].path)
    assert sch == tmp_path / f"{PROJECT_ID}.kicad_sch" and pcb == tmp_path / f"{PROJECT_ID}.kicad_pcb"
    gerber = ir.artifacts[ArtifactKind.GERBER]
    assert Path(gerber.path).name == f"{PROJECT_ID}-job.gbrjob" and Path(gerber.path).parent == tmp_path / "gerber"
    assert len(gerber.files) == 10 and all(Path(f).is_file() for f in gerber.files)
    assert [Path(f).name for f in ir.artifacts[ArtifactKind.DRILL].files] == [f"{PROJECT_ID}.drl"]

    # tool evidence: ERC/DRC ran on exactly these files, parity was really evaluated against this schematic
    erc, drc = ir.validation.latest("kicad.erc"), ir.validation.latest("kicad.drc")
    assert erc.tool == "kicad-cli" and erc.artifact_hash == ir.artifacts[ArtifactKind.SCHEMATIC].content_hash
    assert erc.details["errors"] == [] and erc.details["warnings"] == []
    assert drc.artifact_hash == ir.artifacts[ArtifactKind.PCB].content_hash
    assert drc.details["schematic_parity_checked"] is True
    assert drc.details["schematic_hash"] == ir.artifacts[ArtifactKind.SCHEMATIC].content_hash
    assert drc.details["schematic_parity"] == [] and drc.details["unconnected_items"] == []
    assert "error" in drc.details["included_severities"]
    report = json.loads(Path(drc.evidence[0].path).read_text(encoding="utf-8"))
    assert report["schematic_parity"] == [] and report["source"] == pcb.name
    for check in ("mfg.gerber", "mfg.drill"):
        res = ir.validation.latest(check)
        assert res.status is S.PASS and res.tool == "mfg.output_check"
        assert res.artifact_hash == ir.artifacts[ArtifactKind.GERBER if check == "mfg.gerber" else ArtifactKind.DRILL].content_hash

    # independent review: the slice is proven, the rest is honestly NOT_VERIFIED
    review = _review(ir, tmp_path, ctx.tools)
    for area in PROVEN_AREAS:
        assert review[area].status is S.PASS, f"{area}: {review[area].message}"
    assert review[ReviewArea.SPICE_VS_REQUIREMENTS].status is S.NOT_VERIFIED
    assert review[ReviewArea.REGULATORY_PROVENANCE].status is S.NOT_VERIFIED
    assert review[ReviewArea.MANUFACTURING_CAPABILITIES].status is S.NOT_VERIFIED
    assert not [r for r in review.values() if r.status is S.FAIL]

    # release: never on missing evidence
    release = state.outcomes[-1]
    assert release.stage == Stage.RELEASE and release.status is S.NOT_VERIFIED
    assert release.message.startswith("not releasable: overall validation is NOT_VERIFIED")
    assert "spice" in release.message and "regulatory.research" in release.message
    assert ir.validation.overall() is S.NOT_VERIFIED


def test_compiled_artifacts_are_byte_deterministic_across_runs(tmp_path: Path):
    a = _routed_ir(tmp_path / "a")
    b = _routed_ir(tmp_path / "b")
    Orchestrator(_context(tmp_path / "a")).run(a, stop_after=Stage.PCB)
    Orchestrator(_context(tmp_path / "b")).run(b, stop_after=Stage.PCB)
    for kind in (ArtifactKind.SCHEMATIC, ArtifactKind.PCB):
        assert Path(a.artifacts[kind].path).read_bytes() == Path(b.artifacts[kind].path).read_bytes()
        assert a.artifacts[kind].content_hash == b.artifacts[kind].content_hash


# --------------------------------------------------------------------------- staleness + repair


def test_design_change_is_detected_and_repaired_by_regeneration_only(tmp_path: Path):
    ir, state, ctx = _run(tmp_path)
    assert state.outcomes[-1].status is S.NOT_VERIFIED
    before = {k: v.content_hash for k, v in ir.artifacts.items()}

    add_r3_vout_to_gnd(ir)
    changed_hash = ir.content_hash()

    review = _review(ir, tmp_path, ctx.tools)
    for area in (ReviewArea.IR_VS_SCHEMATIC, ReviewArea.IR_VS_PCB, ReviewArea.PCB_VS_BOM, ReviewArea.PCB_VS_CPL, ReviewArea.MANUFACTURING_OUTPUTS):
        assert review[area].status is S.FAIL and review[area].details["repair"] == "regenerate", area
    assert review[ReviewArea.PCB_VS_BOM].details["only_in_ir"] == ["R3"]
    # ERC/DRC ran on the (still current) old artifacts, so those reports are not stale *yet*
    assert review[ReviewArea.ERC].status is S.PASS and review[ReviewArea.DRC].status is S.PASS

    outcome = RepairLoop(tools=ctx.tools).run(ir, tmp_path)
    print(outcome.summary())
    for a in outcome.actions:
        print(f"  {a.strategy:<28} {a.description:<60} ok={a.succeeded} {a.error or ''}")
    assert outcome.stopped_reason == "all failures resolved"
    assert outcome.unresolved == []
    assert all(a.succeeded for a in outcome.actions)
    assert all(a.ir_hash_before == changed_hash == a.ir_hash_after for a in outcome.actions)  # repairs never touch the IR
    assert ir.content_hash() == changed_hash
    descriptions = [a.description for a in outcome.actions]
    assert descriptions.count("re-run kicad.drc") == 1  # review.drc and review.schematic_vs_pcb ask for the same re-run
    assert "re-run kicad.erc" in descriptions
    assert any(d.startswith("regenerate kicad_sch") for d in descriptions)
    assert any(d.startswith("regenerate gerber, drill") for d in descriptions)
    assert 1 < outcome.iterations <= 3

    final = {r.check_id: r for r in outcome.final_review.results}
    for area in PROVEN_AREAS:
        assert final[area].status is S.PASS, f"{area}: {final[area].message}"
    for kind, old in before.items():
        art = ir.artifacts[kind]
        assert art.generated_from_ir_hash == changed_hash and art.matches_disk()
        assert art.content_hash != old, f"{kind} was not regenerated"
    # the regenerated design is real: R3 is in the schematic, the board, the BOM and the CPL
    assert "R3" in Path(ir.artifacts[ArtifactKind.BOM].path).read_text(encoding="utf-8")
    assert "R3" in Path(ir.artifacts[ArtifactKind.CPL].path).read_text(encoding="utf-8")
    assert '(property "Reference" "R3"' in Path(ir.artifacts[ArtifactKind.PCB].path).read_text(encoding="utf-8")
    drc = ir.validation.latest("kicad.drc")
    assert drc.status is S.PASS and drc.details["errors"] == [] and drc.details["warnings"] == []
    assert drc.details["schematic_parity_checked"] and drc.details["schematic_hash"] == ir.artifacts[ArtifactKind.SCHEMATIC].content_hash
    assert ir.validation.latest("kicad.erc").artifact_hash == ir.artifacts[ArtifactKind.SCHEMATIC].content_hash


def test_missing_sibling_schematic_never_passes_parity(tmp_path: Path):
    """Parity is only claimed when kicad-cli really evaluated it; a lone board gets NOT_VERIFIED, not PASS."""
    ir, _, ctx = _run(tmp_path)
    moved = tmp_path / "elsewhere" / f"{PROJECT_ID}.kicad_sch"
    moved.parent.mkdir()
    moved.write_bytes(Path(ir.artifacts[ArtifactKind.SCHEMATIC].path).read_bytes())
    Path(ir.artifacts[ArtifactKind.SCHEMATIC].path).unlink()
    ir.artifacts[ArtifactKind.SCHEMATIC].path = str(moved)

    from ai_eda.repair.strategies import RerunTool
    from ai_eda.ir import ValidationResult

    finding = ValidationResult(check_id=ReviewArea.DRC, status=S.FAIL, details={"repair": "rerun_tool", "tool_check": "kicad.drc"})
    action = RerunTool().apply(ir, finding, tmp_path, ctx.tools)
    assert action.succeeded
    drc = ir.validation.latest("kicad.drc")
    assert drc.status is S.PASS and drc.details["schematic_parity_checked"] is False
    assert "elsewhere" in drc.details["schematic_parity_reason"]
    review = _review(ir, tmp_path, ctx.tools)
    assert review[ReviewArea.SCHEMATIC_VS_PCB].status is S.NOT_VERIFIED
    assert "not evaluated" in review[ReviewArea.SCHEMATIC_VS_PCB].message
