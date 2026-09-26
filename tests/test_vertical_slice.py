"""The vertical slice, end to end, against the real kicad-cli.

IR (divider + 3-pin header, naive tracks, SPICE bindings + simulation setup)
-> Orchestrator.run:
  CALCULATION PASS -> SPICE (ngspice.dll: op + dc, expectations vs the
  calculator) PASS -> SCHEMATIC PASS -> ERC PASS -> PCB PASS -> DRC
  (+ schematic parity) PASS -> gerber + drill export + format checks PASS
  -> independent review -> RELEASE not PASS (regulatory research and fab
  capability are NOT_VERIFIED, which is the correct verdict for a design
  nobody researched).

Then the staleness scenario: the design changes after the run (a component
with its nets, placement and tracks is added), the reviewer flags every
derived artifact as stale, and the repair loop regenerates them and re-runs
ERC / DRC / SPICE / output checks until the review is clean again - without
ever touching the IR.

Variant chosen for the staleness scenario: the appended component is wired
into the design (R3 = 10k from VOUT to GND, placed and routed so the naive
router stays DRC-clean, bound in SPICE) and R1 is retuned to 5k so the
divider still meets the unchanged 6 V requirement (R2 || R3 = 5k), with the
expectation nominals recomputed by the calculators for the new divider, so
the regenerated artifacts are *valid* and the loop is expected to converge
to PASS on the artifact / tool checks. This is deterministic: every step is
a pure function of the IR plus kicad-cli / ngspice, and the layout was
validated with kicad-cli 10.0.6. (The alternative - leaving the new
component unconnected and asserting the loop reports the resulting ERC/DRC
failures as unresolved - would also be honest, but converging to PASS
proves more of the repair machinery; ``tests/test_simulation_stage.py``
covers the honest SPICE failure after a design change, and
``tests/test_spice_findings_regressions.py`` the case where the nominals are
recalculated but the requirement is not - which the reviewer refuses.)
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
    SpiceBinding,
    SpiceDevice,
    ValidationStatus,
    authoritative,
)
from ai_eda.repair import RepairLoop
from ai_eda.review import IndependentReviewer, ReviewArea
from ai_eda.compilers.pcb import design_rules
from ai_eda.tools.calc import parallel_resistance, recompute_parameters, voltage_divider_output
from ai_eda.tools.kicad import KicadCli, KicadLibrary
from ai_eda.tools.kicad.cli import PROJECT_RULES_MEASURED_VERSIONS
from ai_eda.tools.routing import route_naive
from ai_eda.tools.spice import NgspiceShared
from ai_eda.workflow import Orchestrator, PipelineState, Stage
from tests.conftest import DS, make_component
from tests.fixtures_kicad import HAND_PLACED, PROJECT_ID, divider_with_connector_ir

LIB = KicadLibrary()
HAS_LIBS = LIB.footprint_file("Resistor_SMD", "R_0603_1608Metric") is not None and LIB.symbol_file("Device") is not None
kicad = KicadCli()
ngspice = NgspiceShared()
pytestmark = pytest.mark.skipif(not (kicad.available() and HAS_LIBS and ngspice.available()), reason="kicad-cli / KiCad libraries / ngspice.dll not installed")

S = ValidationStatus

#: review areas the vertical slice must prove with real evidence
PROVEN_AREAS = [
    ReviewArea.IR_VS_SCHEMATIC,
    ReviewArea.IR_VS_PCB,
    ReviewArea.SCHEMATIC_VS_PCB,
    ReviewArea.PCB_VS_BOM,
    ReviewArea.PCB_VS_CPL,
    ReviewArea.MANUFACTURING_OUTPUTS,
    ReviewArea.CALCULATIONS_VS_DESIGN,
    ReviewArea.SPICE_VS_REQUIREMENTS,
    ReviewArea.ERC,
    ReviewArea.DRC,
]


def _context(tmp_path: Path) -> AgentContext:
    return AgentContext(
        workdir=tmp_path,
        tools={"kicad_cli": kicad, "kicad_library": LIB, "spice": ngspice},
        answers={"application": "bench voltage divider", "jurisdiction": "EU"},
    )


def _blocking(release_message: str) -> list[str]:
    """The check ids the RELEASE stage names as blocking: ``... is NOT_VERIFIED (a, b, c)``."""
    return release_message[release_message.index("(") + 1 : release_message.rindex(")")].split(", ")


def _routed_ir(tmp_path: Path) -> CircuitIR:
    ir = divider_with_connector_ir(tmp_path, LIB)
    ir.pcb.tracks = route_naive(ir, LIB)  # copper of its own: the PLACEMENT stage then leaves the board alone (it never replaces copper)
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
    """Append R3 (10k, VOUT -> GND) and retune R1 to 5k, with placement, re-routed tracks, SPICE bindings and recalculated nominals.

    R2 || R3 = 5k and R1 = 5k, so VOUT stays 12 V * 5k / 10k = 6 V and the unchanged requirements
    (``req.v_out`` 6 V, ``req.v_out_half`` 3 V) still hold; the expectation nominals are the calculators'
    outputs for the changed divider (``v_out``, ``v_out_mid`` re-derived through ``r2_eff``), so after
    regeneration + re-simulation the SPICE review is expected to PASS again - a valid design change.
    """
    r3 = make_component("R3", "10k")
    r3.electrical["resistance"] = authoritative(10_000.0, DS, "ohm")
    r3.spice = SpiceBinding(device=SpiceDevice.R, value=r3.electrical["resistance"], provenance=Provenance(kind=ProvenanceKind.AUTHORITATIVE, source=DS))
    ir.components.append(r3)
    r1 = ir.component("R1")
    r1.value = "5k"
    r1.electrical["resistance"] = authoritative(5_000.0, DS, "ohm")
    r1.spice.value = r1.electrical["resistance"]  # the compiler refuses a binding that disagrees with electrical.resistance
    ir.net("VOUT").pins.append(PinRef(component_ref="R3", pin_number="1"))
    ir.net("GND").pins.append(PinRef(component_ref="R3", pin_number="2"))
    ir.pcb.placements.append(Placement(component_ref="R3", x_mm=20.0, y_mm=11.08, rotation_deg=270.0, side=BoardSide.TOP, provenance=HAND_PLACED))
    ir.pcb.tracks = route_naive(ir, LIB)
    p = ir.parameters
    p["r1"] = r1.electrical["resistance"]
    p["r3"] = r3.electrical["resistance"]
    p["r2_eff"] = parallel_resistance(p["r2"], p["r3"], ("r2", "r3"))
    p["v_out"] = voltage_divider_output(p["v_in"], p["r1"], p["r2_eff"], ("v_in", "r1", "r2_eff"))
    p["v_out_mid"] = voltage_divider_output(p["v_in_mid"], p["r1"], p["r2_eff"], ("v_in_mid", "r1", "r2_eff"))
    for exp in ir.simulation.expectations:
        exp.nominal = p["v_out"] if exp.id == "v_out" else p["v_out_mid"]
    assert p["v_out"].value == 6.0 and p["v_out_mid"].value == 3.0 and recompute_parameters(ir).status is S.PASS


# --------------------------------------------------------------------------- the run


def test_pipeline_runs_the_slice_with_real_tools(tmp_path: Path):
    ir, state, ctx = _run(tmp_path)
    assert not state.blocked
    outcomes = {o.stage: o for o in state.outcomes}
    assert [o.stage for o in state.outcomes] == list(Stage)

    for stage in (Stage.CALCULATION, Stage.SPICE, Stage.SCHEMATIC, Stage.ERC, Stage.PCB, Stage.DRC, Stage.MANUFACTURING_OUTPUTS):
        assert outcomes[stage].status is S.PASS, f"{stage}: {outcomes[stage].message}"
    # v_out, v_out_mid and the two expectation nominals that are copies of them (a JSON round trip makes them independent)
    assert outcomes[Stage.CALCULATION].message == "4 value(s) recomputed"
    assert outcomes[Stage.SPICE].message.startswith("2 analysis(es) run [op (op, 1 pt), dc_vin (dc vvin 0 12 1, 13 pt)], 2 expectation(s): 2 PASS")
    assert "re-validated: " in outcomes[Stage.SPICE].message and "domain.analog.bias PASS" in outcomes[Stage.SPICE].message
    assert outcomes[Stage.DRC].message.startswith("0 error(s), 0 warning(s); schematic parity: 0 issue(s)")
    assert "mfg.gerber PASS" in outcomes[Stage.MANUFACTURING_OUTPUTS].message
    assert "mfg.drill PASS" in outcomes[Stage.MANUFACTURING_OUTPUTS].message

    # artifacts: all nine, all fresh, all on disk, schematic and board are siblings with the project stem
    kinds = {ArtifactKind.SPICE_NETLIST, ArtifactKind.SPICE_RESULT, ArtifactKind.SCHEMATIC, ArtifactKind.PCB, ArtifactKind.BOM, ArtifactKind.CPL, ArtifactKind.GERBER, ArtifactKind.DRILL,
             ArtifactKind.KICAD_PROJECT}
    assert kinds <= set(ir.artifacts)
    ir_hash = ir.content_hash()
    for kind in kinds:
        art = ir.artifacts[kind]
        assert art.generated_from_ir_hash == ir_hash and art.matches_disk(), kind
    sch, pcb = Path(ir.artifacts[ArtifactKind.SCHEMATIC].path), Path(ir.artifacts[ArtifactKind.PCB].path)
    assert sch == tmp_path / f"{PROJECT_ID}.kicad_sch" and pcb == tmp_path / f"{PROJECT_ID}.kicad_pcb"
    assert Path(ir.artifacts[ArtifactKind.KICAD_PROJECT].path) == tmp_path / f"{PROJECT_ID}.kicad_pro"  # no fab limits in this IR: empty rules, KiCad defaults
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
    # ERC and DRC ran beside the fresh project file and did not rewrite it
    for res in (erc, drc):
        assert res.details["project_present"] is True and res.details["project_rewritten"] is False and res.details["project_hash"] == ir.artifacts[ArtifactKind.KICAD_PROJECT].content_hash
    assert drc.details["design_rules"] == {} and ir.artifacts[ArtifactKind.KICAD_PROJECT].matches_disk()
    report = json.loads(Path(drc.evidence[0].path).read_text(encoding="utf-8"))
    assert report["schematic_parity"] == [] and report["source"] == pcb.name
    for check in ("mfg.gerber", "mfg.drill"):
        res = ir.validation.latest(check)
        assert res.status is S.PASS and res.tool == "mfg.output_check"
        assert res.artifact_hash == ir.artifacts[ArtifactKind.GERBER if check == "mfg.gerber" else ArtifactKind.DRILL].content_hash
    # SPICE evidence: the netlist that ran is the fresh artifact, results.json + rawfiles are on disk, the numbers are ngspice's
    spice = ir.validation.latest("spice")
    assert spice.status is S.PASS and spice.tool == "ngspice-shared" and spice.tool_version == "ngspice-46"
    assert spice.artifact_hash == ir.artifacts[ArtifactKind.SPICE_NETLIST].content_hash
    assert all(Path(e.path).is_file() and e.content_hash == ir.artifacts[ArtifactKind.SPICE_RESULT].disk_hash() for e in spice.evidence if Path(e.path).name == "results.json")
    assert abs(ir.validation.latest("spice.v_out").details["measured"] - 6.0) < 1e-9
    assert abs(ir.validation.latest("spice.v_out_mid").details["measured"] - 3.0) < 1e-9
    assert ir.validation.latest("domain.analog.bias").status is S.PASS
    assert ir.validation.latest("calc.recompute").status is S.PASS

    # independent review: the slice is proven, the rest is honestly NOT_VERIFIED
    review = _review(ir, tmp_path, ctx.tools)
    for area in PROVEN_AREAS:
        assert review[area].status is S.PASS, f"{area}: {review[area].message}"
    assert review[ReviewArea.REGULATORY_PROVENANCE].status is S.NOT_VERIFIED
    assert review[ReviewArea.MANUFACTURING_CAPABILITIES].status is S.NOT_VERIFIED
    assert not [r for r in review.values() if r.status is S.FAIL]
    # parts / regulatory tracks without a document archive in this context: nothing fetched, nothing verified, nothing upgraded
    # (the online counterpart against a loopback fake is tests/test_parts_regulatory_e2e.py)
    for ref in ("R1", "R2", "J1"):
        existence = ir.validation.latest(f"component.existence.{ref}")
        assert existence.status is S.NOT_VERIFIED and existence.tool == "parts.existence" and "no document archive" in existence.message
    assert ir.validation.latest("ir.component_provenance").status is S.NOT_VERIFIED  # tagged authoritative, not grounded in an archived datasheet
    assert review[ReviewArea.COMPONENT_PROVENANCE].status is S.NOT_VERIFIED and "not fully verified" in review[ReviewArea.COMPONENT_PROVENANCE].message
    assert ir.validation.latest("regulatory.compliance").status is S.NOT_VERIFIED
    assert "compliance not assessed" in review[ReviewArea.REGULATORY_PROVENANCE].message

    # release: never on missing evidence - and the message names exactly what is missing
    release = state.outcomes[-1]
    assert release.stage == Stage.RELEASE and release.status is S.NOT_VERIFIED
    assert release.message.startswith("not releasable: overall validation is NOT_VERIFIED")
    blocking = _blocking(release.message)
    assert {"regulatory.research", "regulatory.compliance", "mfg.capability", "review.regulatory_provenance", "review.manufacturing_capabilities",
            "review.component_provenance", "component.existence.R1"} <= set(blocking)
    assert not [b for b in blocking if "spice" in b or "calc" in b or "analog" in b], blocking
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
    for area in (ReviewArea.IR_VS_SCHEMATIC, ReviewArea.IR_VS_PCB, ReviewArea.PCB_VS_BOM, ReviewArea.PCB_VS_CPL, ReviewArea.MANUFACTURING_OUTPUTS, ReviewArea.SPICE_VS_REQUIREMENTS):
        assert review[area].status is S.FAIL and review[area].details["repair"] == "regenerate", area
    assert review[ReviewArea.PCB_VS_BOM].details["only_in_ir"] == ["R3"]
    assert review[ReviewArea.SPICE_VS_REQUIREMENTS].details["artifact"] == ArtifactKind.SPICE_NETLIST
    # ERC ran on the (still current) old schematic, so that report is not stale *yet*; the DRC report read the project file,
    # which the design change made stale: the review asks for the project file to be regenerated (the DRC re-run follows)
    assert review[ReviewArea.ERC].status is S.PASS
    assert review[ReviewArea.DRC].status is S.FAIL and review[ReviewArea.DRC].details["repair"] == "regenerate" and review[ReviewArea.DRC].details["artifacts"] == ["kicad_pro"]

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
    assert descriptions.count("regenerate spice_netlist from IR") == 1 and descriptions.count("re-run spice") == 1
    assert descriptions.index("regenerate spice_netlist from IR") < descriptions.index("re-run spice")
    assert any(d.startswith("regenerate kicad_sch") for d in descriptions)
    assert any(d.startswith("regenerate gerber, drill") for d in descriptions)
    assert "regenerate kicad_pro from IR" in descriptions
    assert 1 < outcome.iterations <= 3

    final = {r.check_id: r for r in outcome.final_review.results}
    for area in PROVEN_AREAS:
        assert final[area].status is S.PASS, f"{area}: {final[area].message}"
    for kind, old in before.items():
        art = ir.artifacts[kind]
        assert art.generated_from_ir_hash == changed_hash and art.matches_disk()
        if kind is ArtifactKind.KICAD_PROJECT:
            assert art.content_hash == old  # regenerated from the changed IR, but the rules did not change: same bytes
        else:
            assert art.content_hash != old, f"{kind} was not regenerated"
    # the regenerated design is real: R3 is in the schematic, the board, the BOM, the CPL and the netlist ngspice ran
    assert "R3" in Path(ir.artifacts[ArtifactKind.BOM].path).read_text(encoding="utf-8")
    assert "R3" in Path(ir.artifacts[ArtifactKind.CPL].path).read_text(encoding="utf-8")
    assert '(property "Reference" "R3"' in Path(ir.artifacts[ArtifactKind.PCB].path).read_text(encoding="utf-8")
    netlist_text = Path(ir.artifacts[ArtifactKind.SPICE_NETLIST].path).read_text(encoding="utf-8")
    assert "R3 VOUT 0 10k" in netlist_text and "R1 VIN VOUT 5k" in netlist_text
    assert "5k" in Path(ir.artifacts[ArtifactKind.BOM].path).read_text(encoding="utf-8")  # the BOM ships what ngspice simulated
    v_out = ir.validation.latest("spice.v_out")
    assert v_out.status is S.PASS and abs(v_out.details["measured"] - 6.0) < 1e-9 and v_out.artifact_hash == ir.artifacts[ArtifactKind.SPICE_NETLIST].content_hash
    assert abs(ir.validation.latest("spice.v_out_mid").details["measured"] - 3.0) < 1e-9
    drc = ir.validation.latest("kicad.drc")
    assert drc.status is S.PASS and drc.details["errors"] == [] and drc.details["warnings"] == []
    assert drc.details["schematic_parity_checked"] and drc.details["schematic_hash"] == ir.artifacts[ArtifactKind.SCHEMATIC].content_hash
    assert drc.details["project_hash"] == ir.artifacts[ArtifactKind.KICAD_PROJECT].content_hash
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


# --------------------------------------------------------------------------- fab capability (needs the Windows measurement)


CAPABILITY_URL = "https://fab.example.com/capabilities"


def _capability_file(tmp_path: Path) -> Path:
    """A capability file the slice satisfies (0.25 mm naive tracks, no vias, 2 layers), served by the loopback fake."""
    limits = [
        {"key": "min_track_width_mm", "value": 0.127, "unit": "mm", "page": 1, "quote": "Min trace width 0.127 mm"},
        {"key": "min_clearance_mm", "value": 0.127, "unit": "mm", "page": 1, "quote": "Min spacing 0.127 mm"},
        {"key": "min_via_drill_mm", "value": 0.3, "unit": "mm", "page": 1, "quote": "Min via hole size 0.3 mm"},
        {"key": "min_via_diameter_mm", "value": 0.5, "unit": "mm", "page": 1, "quote": "Min via diameter 0.5 mm"},
        {"key": "layer_count_options", "value": [1, 2, 4], "unit": None, "page": 1, "quote": "Layers 1, 2, 4"},
        {"key": "copper_weight_oz", "value": 1, "unit": "oz", "page": 1, "quote": "Copper weight 1 oz"},
        {"key": "board_thickness_mm", "value": 1.6, "unit": "mm", "page": 1, "quote": "Board thickness 1.6 mm"},
    ]
    p = tmp_path / "fab.json"
    p.write_text(json.dumps({"fab": "Example Fab", "source": {"url": CAPABILITY_URL, "title": "Example Fab capabilities", "authority": "Example Fab"}, "limits": limits}), encoding="utf-8")
    return p


CAPABILITY_HTML = (
    "<html><head><title>Example Fab capabilities</title></head><body><table>"
    "<tr><td>Min trace width</td><td>0.127 mm</td></tr><tr><td>Min spacing</td><td>0.127 mm</td></tr>"
    "<tr><td>Min via hole size</td><td>0.3 mm</td></tr><tr><td>Min via diameter</td><td>0.5 mm</td></tr>"
    "<tr><td>Layers</td><td>1, 2, 4</td></tr><tr><td>Copper weight</td><td>1 oz</td></tr><tr><td>Board thickness</td><td>1.6 mm</td></tr>"
    "</table></body></html>"
)


@pytest.mark.skipif(not (kicad.available() and kicad.version() in PROJECT_RULES_MEASURED_VERSIONS),
                    reason="kicad-cli's application of sibling .kicad_pro rules is not measured for this version (PROJECT_RULES_MEASURED_VERSIONS)")
def test_fab_capability_limits_become_drc_rules_and_pass(tmp_path: Path):
    """With a grounded capability file the project file carries the fab minimums, DRC runs with them, mfg.capability PASSes on that
    evidence; a track narrowed below the limit is a fab_capability_shortfall (human) and the release FAILs."""
    from ai_eda.security import ApprovalGate
    from ai_eda.workflow import open_session
    from tests.fake_sources import FakeSources

    cap = _capability_file(tmp_path)
    with FakeSources() as fake:
        fake.add_html(CAPABILITY_URL, CAPABILITY_HTML)
        ir = _routed_ir(tmp_path)
        session = open_session(workdir=tmp_path, ir=ir, library=LIB, online=True, fab_capability=cap, gate=ApprovalGate(), client=fake.client())
        try:
            ctx = AgentContext(workdir=tmp_path, tools={"kicad_cli": kicad, "kicad_library": LIB, "spice": ngspice, **session.tools()},
                               answers={"application": "bench voltage divider", "jurisdiction": "EU"})
            state = Orchestrator(ctx).run(ir)
        finally:
            session.close()
        for o in state.outcomes:
            print(f"{o.stage:<24} {o.status:<14} {o.message}")
        assert not state.blocked and state.outcome(Stage.FAB_CAPABILITY).status is S.PASS
        assert ir.pcb.manufacturing.min_track_width_mm.value == 0.127 and ir.pcb.manufacturing.min_clearance_mm.value == 0.127
        pro = ir.artifacts[ArtifactKind.KICAD_PROJECT]
        assert Path(pro.path) == tmp_path / f"{PROJECT_ID}.kicad_pro" and pro.matches_disk() and pro.generated_from_ir_hash == ir.content_hash()
        drc = ir.validation.latest("kicad.drc")
        assert drc.status is S.PASS and drc.details["project_hash"] == pro.content_hash
        assert drc.details["design_rules"] == design_rules(ir) == {"min_track_width": 0.127, "min_clearance": 0.127, "min_via_diameter": 0.5, "min_through_hole_diameter": 0.3}
        cap_res = ir.validation.latest("mfg.capability")
        assert cap_res.status is S.PASS, cap_res.message
        assert cap_res.ir_hash == ir.content_hash() and cap_res.details["drc_artifact_hash"] == ir.artifacts[ArtifactKind.PCB].content_hash
        assert "(thickness 1.6)" in Path(ir.artifacts[ArtifactKind.PCB].path).read_text(encoding="utf-8")
        review = _review(ir, tmp_path, ctx.tools)
        assert review[ReviewArea.MANUFACTURING_CAPABILITIES].status is S.PASS and review[ReviewArea.DRC].status is S.PASS
        assert state.outcome(Stage.RELEASE).status is S.NOT_VERIFIED and "mfg.capability" not in _blocking(state.outcome(Stage.RELEASE).message)

        # a track below the fab minimum: the IR comparison FAILs (human), DRC with the fab rules FAILs too, the release FAILs
        ir.pcb.tracks[0].width_mm = 0.1
        session = open_session(workdir=tmp_path, ir=ir, library=LIB, online=False, fab_capability=cap, gate=ApprovalGate())
        try:
            ctx = AgentContext(workdir=tmp_path, tools={"kicad_cli": kicad, "kicad_library": LIB, "spice": ngspice, **session.tools()},
                               answers={"application": "bench voltage divider", "jurisdiction": "EU"})
            state = Orchestrator(ctx).run(ir)
        finally:
            session.close()
        cap_res = ir.validation.latest("mfg.capability")
        assert cap_res.status is S.FAIL and cap_res.details["repair"] == "fab_capability_shortfall" and "track[0:" in cap_res.message
        assert ir.validation.latest("kicad.drc").status is S.FAIL
        assert state.outcome(Stage.RELEASE).status is S.FAIL and "mfg.capability" in state.outcome(Stage.RELEASE).message
