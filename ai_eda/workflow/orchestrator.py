"""Stage runner.

Each stage is a small function ``(ir, ctx) -> StageOutcome``. The
orchestrator runs them in :data:`STAGE_ORDER`, stops at the first stage that
is blocked on user input, and records every outcome so the GUI can show
where the design is and why.

What the orchestrator does *not* do: it never places, routes or otherwise
changes the design to make a stage pass. It compiles what the IR contains
(``ir.pcb.tracks`` included) and lets the tools judge it. A stage whose input
does not exist yet (no components, no ``ir.pcb``, no board to export) is
NOT_VERIFIED; a stage whose input is inconsistent (pins that do not match
the library, a pin in no net, an unverified footprint) is FAIL with the
compiler's message. Any other exception propagates - it is a defect, not a
verdict.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from pydantic import BaseModel, Field

from ai_eda.agents import (
    AgentContext,
    AgentResult,
    IRProposal,
    CircuitDesignAgent,
    ComponentAgent,
    ManufacturingAgent,
    PCBAgent,
    RegulatoryAgent,
    RepairAgent,
    RequirementAgent,
    ReviewAgent,
    SimulationAgent,
)
from ai_eda.compilers import (
    BOMCompiler,
    CPLCompiler,
    CompileContext,
    DrillExporter,
    GerberExporter,
    PCBCompiler,
    SchematicCompiler,
    SpiceNetlistCompiler,
)
from ai_eda.errors import CompileError, NothingToCompileError, ToolUnavailableError
from ai_eda.ir import ArtifactKind, CircuitIR, MissingInformation, ValidationResult, ValidationStatus, worst_status
from ai_eda.tools.calc import recompute_parameters
from ai_eda.tools.kicad.cli import KicadCli, run_drc_for, run_erc_for
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.manufacturing.outputs import check_output_artifact
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.workflow.stages import STAGE_ORDER, Stage


class StageOutcome(BaseModel):
    stage: Stage
    status: ValidationStatus
    message: str = ""
    questions: list[MissingInformation] = Field(default_factory=list)
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PipelineState(BaseModel):
    outcomes: list[StageOutcome] = Field(default_factory=list)
    current: Stage | None = None
    blocked: bool = False

    @property
    def open_questions(self) -> list[MissingInformation]:
        return [q for o in self.outcomes for q in o.questions if q.required]

    def outcome(self, stage: Stage) -> StageOutcome | None:
        for o in self.outcomes:
            if o.stage == stage:
                return o
        return None


StageFn = Callable[[CircuitIR, AgentContext], StageOutcome]

#: artifact kinds produced by the manufacturing-outputs stage, in export order
MANUFACTURING_EXPORTS: tuple[ArtifactKind, ...] = (ArtifactKind.GERBER, ArtifactKind.DRILL)


class Orchestrator:
    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx
        self.ctx.tools.setdefault(
            "compilers",
            {
                ArtifactKind.SCHEMATIC: SchematicCompiler(),
                ArtifactKind.PCB: PCBCompiler(),
                ArtifactKind.BOM: BOMCompiler(),
                ArtifactKind.CPL: CPLCompiler(),
                ArtifactKind.SPICE_NETLIST: SpiceNetlistCompiler(),
                ArtifactKind.GERBER: GerberExporter(),
                ArtifactKind.DRILL: DrillExporter(),
            },
        )
        # One library instance for the schematic / PCB compilers (parsed libraries are cached per instance).
        self.ctx.tools.setdefault("kicad_library", KicadLibrary())
        self.stages: dict[Stage, StageFn] = {
            Stage.REQUIREMENT_ANALYSIS: self._agent_stage(RequirementAgent()),
            Stage.MISSING_INFORMATION: self._missing_information,
            Stage.REGULATORY_RESEARCH: self._agent_stage(RegulatoryAgent()),
            Stage.ARCHITECTURE: self._agent_stage(CircuitDesignAgent()),
            Stage.COMPONENT_SELECTION: self._agent_stage(ComponentAgent()),
            Stage.IR_BUILD: self._ir_validate,
            Stage.CALCULATION: self._calculation,
            Stage.SPICE: self._agent_stage(SimulationAgent()),
            Stage.SCHEMATIC: self._compile_stage(ArtifactKind.SCHEMATIC),
            Stage.ERC: self._kicad_check("kicad.erc"),
            Stage.PCB: self._compile_stage(ArtifactKind.PCB),
            Stage.DRC: self._kicad_check("kicad.drc"),
            Stage.MANUFACTURABILITY: self._agent_stage(ManufacturingAgent()),
            Stage.MANUFACTURING_OUTPUTS: self._manufacturing_outputs,
            Stage.INDEPENDENT_REVIEW: self._agent_stage(ReviewAgent()),
            Stage.REPAIR: self._agent_stage(RepairAgent()),
            Stage.RELEASE: self._release,
        }

    # --- driver ------------------------------------------------------------------

    def run(self, ir: CircuitIR, stop_after: Stage | None = None) -> PipelineState:
        state = PipelineState()
        for stage in STAGE_ORDER:
            state.current = stage
            outcome = self.stages[stage](ir, self.ctx)
            state.outcomes.append(outcome)
            if outcome.status == ValidationStatus.USER_INPUT_REQUIRED:
                state.blocked = True
                break
            if stage == stop_after:
                break
        return state

    # --- stage builders ----------------------------------------------------------

    @staticmethod
    def apply_proposals(ir: CircuitIR, proposals: list[IRProposal]) -> None:
        """Apply agent proposals to the IR.

        This is the *only* place agent output touches the IR, so it is the
        natural hook for user confirmation / GUI diffing later. Only simple
        append/set operations on known targets are supported for now.
        """
        for p in proposals:
            obj: Any = ir
            parts = p.target.split(".")
            for part in parts[:-1]:
                obj = getattr(obj, part)
            leaf = parts[-1]
            if p.operation == "append":
                getattr(obj, leaf).append(p.payload)
            elif p.operation == "set":
                if isinstance(obj, dict):
                    obj[leaf] = p.payload
                else:
                    setattr(obj, leaf, p.payload)
            elif p.operation == "remove":
                seq = getattr(obj, leaf)
                seq[:] = [x for x in seq if x != p.payload]
            else:
                raise ValueError(f"unknown proposal operation {p.operation!r}")

    def _agent_stage(self, agent) -> StageFn:
        def fn(ir: CircuitIR, ctx: AgentContext) -> StageOutcome:
            result: AgentResult = agent.run(ir, ctx)
            self.apply_proposals(ir, result.proposals)
            ir.validation.extend(result.validation)
            notes = list(result.notes)
            if result.blocked_on_user:
                status = ValidationStatus.USER_INPUT_REQUIRED
            elif result.validation:
                status = worst_status(r.status for r in result.validation)
            else:
                # A proposal is not evidence (CLAUDE.md #3): an agent that only proposed (or produced
                # nothing) has verified nothing, whether or not its proposals were applied.
                status = ValidationStatus.NOT_VERIFIED
                if result.proposals:
                    notes.insert(0, f"{len(result.proposals)} proposal(s) applied, nothing verified")
            revalidated = self._revalidate(ir, ctx, {r.check_id for r in result.validation})
            if revalidated:
                notes.append("re-validated: " + ", ".join(f"{r.check_id} {r.status}" for r in revalidated))
            return StageOutcome(stage=Stage(self._stage_of(agent)), status=status, message="; ".join(notes), questions=result.questions)
        return fn

    @staticmethod
    def _revalidate(ir: CircuitIR, ctx: AgentContext, produced: set[str]) -> list[ValidationResult]:
        """Re-run the registered validators that consume a check id this stage just produced.

        A validator such as ``domain.analog.bias`` reads the ``spice`` result;
        at IR_BUILD time that result does not exist yet, so it is evaluated
        again right after the stage that produces it. The stage's own status
        stays the agent's verdict; the re-validation is reported in the message.
        """
        if not produced:
            return []
        out: list[ValidationResult] = []
        vctx = ValidationContext(workdir=ctx.workdir, tools=ctx.tools)
        for v in default_registry.select(ir):
            if v.consumes & produced:
                out.extend(v.validate(ir, vctx))
        ir.validation.extend(out)
        return out

    def _stage_of(self, agent) -> str:
        return {
            "requirement": Stage.REQUIREMENT_ANALYSIS,
            "regulatory": Stage.REGULATORY_RESEARCH,
            "circuit_design": Stage.ARCHITECTURE,
            "component": Stage.COMPONENT_SELECTION,
            "simulation": Stage.SPICE,
            "manufacturing": Stage.MANUFACTURABILITY,
            "review": Stage.INDEPENDENT_REVIEW,
            "repair": Stage.REPAIR,
        }[agent.name]

    def _missing_information(self, ir: CircuitIR, ctx: AgentContext) -> StageOutcome:
        pending = [q for q in ir.requirements.missing if q.required and q.key not in ctx.answers]
        if pending:
            return StageOutcome(stage=Stage.MISSING_INFORMATION, status=ValidationStatus.USER_INPUT_REQUIRED, questions=pending, message=f"{len(pending)} required question(s)")
        return StageOutcome(stage=Stage.MISSING_INFORMATION, status=ValidationStatus.PASS)

    def _ir_validate(self, ir: CircuitIR, ctx: AgentContext) -> StageOutcome:
        results = default_registry.run(ir, ValidationContext(workdir=ctx.workdir, tools=ctx.tools))
        ir.validation.extend(results)
        status = worst_status(r.status for r in results)
        # a validator that needs the user (assumptions, undecided model-inferred requirements) is shown as a
        # question keyed by its check id, so the CLI lists what to answer instead of a bare BLOCKED
        questions = [
            MissingInformation(key=r.check_id, question=r.message, rationale=r.check_id)
            for r in results if r.status == ValidationStatus.USER_INPUT_REQUIRED
        ]
        return StageOutcome(stage=Stage.IR_BUILD, status=status, message=f"{len(results)} validator result(s)", questions=questions)

    @staticmethod
    def _compile(ir: CircuitIR, ctx: AgentContext, kind: ArtifactKind) -> tuple[ValidationStatus, str]:
        """Compile one artifact kind into ``ir.artifacts``; maps compiler refusals to statuses.

        NothingToCompileError / NotImplementedError -> NOT_VERIFIED (the
        input does not exist yet), CompileError -> FAIL (the IR is
        inconsistent); anything else propagates.
        """
        compiler = ctx.tools["compilers"][kind]
        try:
            ir.artifacts[kind] = compiler.compile(ir, CompileContext(workdir=ctx.workdir, tools=ctx.tools))
        except (NothingToCompileError, NotImplementedError) as e:
            ir.artifacts.pop(kind, None)  # an older artifact of this kind would be stale evidence
            return ValidationStatus.NOT_VERIFIED, str(e)
        except CompileError as e:
            ir.artifacts.pop(kind, None)
            return ValidationStatus.FAIL, f"{kind} compile refused: {e}"
        return ValidationStatus.PASS, str(ir.artifacts[kind].path)

    def _compile_stage(self, kind: ArtifactKind) -> StageFn:
        stage = Stage.SCHEMATIC if kind == ArtifactKind.SCHEMATIC else Stage.PCB

        def fn(ir: CircuitIR, ctx: AgentContext) -> StageOutcome:
            status, message = self._compile(ir, ctx, kind)
            return StageOutcome(stage=stage, status=status, message=message)
        return fn

    def _kicad_check(self, check: str) -> StageFn:
        stage = Stage.ERC if check == "kicad.erc" else Stage.DRC
        kind = ArtifactKind.SCHEMATIC if check == "kicad.erc" else ArtifactKind.PCB

        def fn(ir: CircuitIR, ctx: AgentContext) -> StageOutcome:
            kicad = ctx.tools.get("kicad_cli")
            art = ir.artifacts.get(kind)
            if not isinstance(kicad, KicadCli) or not kicad.available():
                return StageOutcome(stage=stage, status=ValidationStatus.NOT_VERIFIED, message="kicad-cli not available")
            if art is None:
                return StageOutcome(stage=stage, status=ValidationStatus.NOT_VERIFIED, message=f"no {kind} to check")
            if check == "kicad.erc":
                res = run_erc_for(ir, kicad, ctx.workdir)
                message = res.message
            else:
                res = run_drc_for(ir, kicad, ctx.workdir)
                if res.details.get("schematic_parity_checked"):
                    message = f"{res.message}; schematic parity: {len(res.details.get('schematic_parity', []))} issue(s)"
                else:
                    message = f"{res.message}; schematic parity not evaluated ({res.details.get('schematic_parity_reason', 'unknown')})"
            ir.validation.add(res)
            return StageOutcome(stage=stage, status=res.status, message=message)
        return fn

    def _manufacturing_outputs(self, ir: CircuitIR, ctx: AgentContext) -> StageOutcome:
        """BOM/CPL from the IR, then gerber + drill from the board via kicad-cli, then the format checks."""
        stage = Stage.MANUFACTURING_OUTPUTS
        compilers = ctx.tools["compilers"]
        cctx = CompileContext(workdir=ctx.workdir, tools=ctx.tools)
        ir.artifacts[ArtifactKind.BOM] = compilers[ArtifactKind.BOM].compile(ir, cctx)
        ir.artifacts[ArtifactKind.CPL] = compilers[ArtifactKind.CPL].compile(ir, cctx)
        if ArtifactKind.PCB not in ir.artifacts:
            return StageOutcome(stage=stage, status=ValidationStatus.NOT_VERIFIED, message="BOM/CPL generated; gerber/drill skipped (no PCB)")
        notes: list[str] = ["BOM/CPL generated"]
        statuses: list[ValidationStatus] = []
        for kind in MANUFACTURING_EXPORTS:
            try:
                status, message = self._compile(ir, ctx, kind)
            except ToolUnavailableError as e:
                return StageOutcome(stage=stage, status=ValidationStatus.NOT_VERIFIED, message=f"{'; '.join(notes)}; {kind} export skipped: {e}")
            if status != ValidationStatus.PASS:
                statuses.append(status)
                notes.append(f"{kind}: {message}")
                continue
            res = check_output_artifact(ir.artifacts[kind], ir)
            res.ir_hash = ir.content_hash()
            ir.validation.add(res)
            statuses.append(res.status)
            notes.append(f"{res.check_id} {res.status}: {res.message}")
        return StageOutcome(stage=stage, status=worst_status(statuses), message="; ".join(notes))

    def _calculation(self, ir: CircuitIR, ctx: AgentContext) -> StageOutcome:
        """Recompute every derived parameter with its registered calculator (``calc.recompute``).

        PASS when all recomputed values match the stored ones, FAIL (human)
        on a mismatch, NOT_VERIFIED when a parameter's tool is not a
        registered calculator or nothing is derived - the stage never
        changes a parameter, it only checks that the IR's numbers are the
        calculators' numbers.
        """
        res = recompute_parameters(ir)
        res.ir_hash = ir.content_hash()
        ir.validation.add(res)
        return StageOutcome(stage=Stage.CALCULATION, status=res.status, message=res.message)

    def _release(self, ir: CircuitIR, ctx: AgentContext) -> StageOutcome:
        overall = ir.validation.overall()
        ok = overall == ValidationStatus.PASS
        if ok:
            return StageOutcome(stage=Stage.RELEASE, status=ValidationStatus.PASS, message="evidence-backed release")
        blocking = sorted(k for k, r in ir.validation.latest_by_check().items() if r.status == overall)
        return StageOutcome(
            stage=Stage.RELEASE,
            status=overall,
            message=f"not releasable: overall validation is {overall} ({', '.join(blocking)})",
        )
