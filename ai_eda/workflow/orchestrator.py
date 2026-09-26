"""Stage runner.

Each stage is a small function ``(ir, ctx) -> StageOutcome``. The
orchestrator runs them in :data:`STAGE_ORDER`, stops at the first stage that
is blocked on user input, and records every outcome so the GUI can show
where the design is and why.

What the orchestrator does *not* do: it never places, routes or otherwise
changes the design itself to make a stage pass. The PLACEMENT stage applies
the ``PCBAgent``'s one deterministic proposal - the grid placement plus the
tracks and vias ``routing.maze`` derived from it (or the placement alone
when a net is unroutable or ``--answer pcb.routing=skip`` was given) - like
any other proposal (through :meth:`Orchestrator.apply_proposals`, before
IR_BUILD so every validator hash is about the placed, routed design and the
``pcb.routing.*`` IR-geometry checks judge that copper there), the
FAB_CAPABILITY stage right after it records the fab limits the
``FabCapabilityAgent`` grounded on the archived vendor page into
``ir.pcb.manufacturing`` (merged, never replacing what the user wrote;
before IR_BUILD for the same reason - the limits feed the ``.kicad_pro``
design rules written in the SCHEMATIC stage and the board thickness), and
the compilers / DRC judge it. Copper the IR already carries is never
replaced by a proposal. The orchestrator compiles what the
IR contains (``ir.pcb.tracks`` included) and lets the tools judge it. A stage whose input
does not exist yet (no components, no ``ir.pcb``, no board to export) is
NOT_VERIFIED; a stage whose input is inconsistent (pins that do not match
the library, a pin in no net, an unverified footprint) is FAIL with the
compiler's message. Any other exception propagates - it is a defect, not a
verdict.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, get_args, get_origin

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from ai_eda.agents import (
    AgentContext,
    AgentResult,
    IRProposal,
    CircuitDesignAgent,
    ComponentAgent,
    FabCapabilityAgent,
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
    ProjectFileCompiler,
    SchematicCompiler,
    SpiceNetlistCompiler,
)
from ai_eda.errors import CompileError, NothingToCompileError, ToolUnavailableError
from ai_eda.ir import ArtifactKind, CircuitIR, MissingInformation, ValidationResult, ValidationStatus, worst_status
from ai_eda.ir.provenance import design_data
from ai_eda.tools.calc import recompute_parameters
from ai_eda.tools.kicad.cli import KicadCli, run_drc_for, run_erc_for
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.manufacturing.outputs import check_output_artifact
from ai_eda.validation import ValidationContext, default_registry
from ai_eda.workflow.stages import STAGE_ORDER, Stage


def _stamp(results: list[ValidationResult], ir: CircuitIR) -> None:
    """Record the IR version the results are about (a result that already carries one keeps it)."""
    if results:
        h = ir.content_hash()
        for r in results:
            r.ir_hash = r.ir_hash or h


def _neutralised_summary(notes: list[str]) -> str:
    """``'N cell(s) neutralised (R1.Value, ...)'`` from compiler notes of the form ``'<where>: <why>; ...'`` (one line per stage)."""
    return f"{len(notes)} cell(s) neutralised ({', '.join(n.split(':', 1)[0] for n in notes)})"


def _field_annotation(model: Any, name: str) -> Any:
    """The declared type of field ``name`` on a pydantic model instance (``Any`` when it is not a model field)."""
    if isinstance(model, BaseModel):
        field = type(model).model_fields.get(name)
        if field is not None and field.annotation is not None:
            return field.annotation
        raise ValueError(f"{type(model).__name__} has no field {name!r}")
    return Any


def _sequence(obj: Any, leaf: str, where: str) -> list:
    seq = obj[leaf] if isinstance(obj, dict) else getattr(obj, leaf)
    if not isinstance(seq, list):
        raise ValueError(f"{where}: {leaf!r} is not a list")
    return seq


def _validated(annotation: Any, payload: Any, where: str) -> Any:
    """``payload`` validated as ``annotation`` (an already-valid model instance passes through unchanged)."""
    if annotation is Any:
        return payload
    try:
        return TypeAdapter(annotation).validate_python(payload)
    except ValidationError as e:
        raise ValueError(f"{where}: payload is not a valid {annotation!r}: {e.errors()[0].get('msg', e) if e.errors() else e}") from e


def _design_view(item: Any) -> Any:
    """The design content of an IR element (two equal designs compare equal whatever their clocks and paths say)."""
    if isinstance(item, BaseModel):
        return design_data(item)
    return item


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

    @property
    def optional_questions(self) -> list[MissingInformation]:
        """Questions a stage asked without stopping the pipeline (regulatory scope answers, model proposals to accept); answerable with ``--answer``."""
        return [q for o in self.outcomes for q in o.questions if not q.required]

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
                ArtifactKind.KICAD_PROJECT: ProjectFileCompiler(),
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
            Stage.PLACEMENT: self._agent_stage(PCBAgent()),
            Stage.FAB_CAPABILITY: self._agent_stage(FabCapabilityAgent()),
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

    def run(self, ir: CircuitIR, stop_after: Stage | None = None, *, state: PipelineState | None = None) -> PipelineState:
        """Run the stages in order into ``state`` (a fresh one by default) and return it.

        A caller that passes its own ``state`` keeps the partial outcomes when
        a stage raises: ``state.current`` is then the stage that died.
        """
        state = PipelineState() if state is None else state
        self._fresh_from = len(ir.validation.results)  # results before this index were carried over from earlier runs
        for stage in STAGE_ORDER:
            state.current = stage
            outcome = self.stages[stage](ir, self.ctx)
            state.outcomes.append(outcome)
            # a required question stops the pipeline even when a FAIL in the same stage outranks
            # USER_INPUT_REQUIRED in the aggregated status: the user is asked, not run past
            if outcome.status == ValidationStatus.USER_INPUT_REQUIRED or any(q.required for q in outcome.questions):
                state.blocked = True
                break
            if stage == stop_after:
                break
        return state

    # --- stage builders ----------------------------------------------------------

    @staticmethod
    def apply_proposals(ir: CircuitIR, proposals: list[IRProposal]) -> None:
        """Apply agent proposals to the design content of the IR.

        This is the *only* place agent output touches the design, so it is the
        natural hook for user confirmation / GUI diffing later. Every payload
        is validated against the type of the field it lands in (the IR's own
        pydantic models: a dict for a ``Topology`` becomes a ``Topology`` or
        raises, a string for ``components`` raises), so a malformed proposal
        can not leave an IR that hashes today and fails to load tomorrow.
        ``remove`` matches by design content (wall-clock ``created_at`` is not
        content) and raises when nothing matched: a removal that silently did
        nothing is worse than one that fails.
        """
        # two phases: every proposal is resolved and validated first, then all are applied - a set of proposals
        # is one logical change, and a bad one must not leave the first half applied (and the hash moved)
        plan: list[Callable[[], None]] = []
        for p in proposals:
            obj: Any = ir
            parts = p.target.split(".")
            for part in parts[:-1]:
                obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
            leaf = parts[-1]
            if isinstance(obj, dict):
                # the parent model's annotation names the value type of this dict (``parameters: dict[str, Traced]``)
                owner: Any = ir
                for part in parts[:-2]:
                    owner = owner[part] if isinstance(owner, dict) else getattr(owner, part)
                annotation = _field_annotation(owner, parts[-2]) if len(parts) >= 2 and isinstance(owner, BaseModel) else Any
                value_type = get_args(annotation)[1] if get_origin(annotation) is dict and len(get_args(annotation)) == 2 else Any
            else:
                annotation = _field_annotation(obj, leaf)
                value_type = annotation
            where = f"proposal {p.description!r} ({p.operation} {p.target})"
            if p.operation == "append":
                seq = _sequence(obj, leaf, where)
                item_type = get_args(annotation)[0] if get_origin(annotation) is list and get_args(annotation) else Any
                item = _validated(item_type, p.payload, where)
                plan.append(lambda seq=seq, item=item: seq.append(item))
            elif p.operation == "set":
                value = _validated(value_type, p.payload, where)
                if isinstance(obj, dict):
                    plan.append(lambda obj=obj, leaf=leaf, value=value: obj.__setitem__(leaf, value))
                else:
                    plan.append(lambda obj=obj, leaf=leaf, value=value: setattr(obj, leaf, value))
            elif p.operation == "remove":
                seq = _sequence(obj, leaf, where)
                item_type = get_args(annotation)[0] if get_origin(annotation) is list and get_args(annotation) else Any
                wanted = _design_view(_validated(item_type, p.payload, where))
                if not any(_design_view(x) == wanted for x in seq):
                    raise ValueError(f"{where}: nothing in {p.target} matches the payload; the removal would have been silent")
                plan.append(lambda seq=seq, wanted=wanted: seq.__setitem__(slice(None), [x for x in seq if _design_view(x) != wanted]))
            else:
                raise ValueError(f"unknown proposal operation {p.operation!r}")
        for step in plan:
            step()

    def _agent_stage(self, agent) -> StageFn:
        def fn(ir: CircuitIR, ctx: AgentContext) -> StageOutcome:
            result: AgentResult = agent.run(ir, ctx)
            self.apply_proposals(ir, result.proposals)
            # The results of an agent that proposes were computed on the IR *before* its proposals, about the part
            # of the design it owns and proposes; they carry no ir_hash (neither hash would be honest) and count as
            # evidence only in the run that produced them (see _release). An agent that proposes nothing and judges
            # the IR as it stands (ManufacturingAgent's mfg.capability) stamps the hash itself.
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
        _stamp(out, ir)
        ir.validation.extend(out)
        return out

    def _stage_of(self, agent) -> str:
        return {
            "requirement": Stage.REQUIREMENT_ANALYSIS,
            "regulatory": Stage.REGULATORY_RESEARCH,
            "circuit_design": Stage.ARCHITECTURE,
            "component": Stage.COMPONENT_SELECTION,
            "pcb": Stage.PLACEMENT,
            "fab_capability": Stage.FAB_CAPABILITY,
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
        _stamp(results, ir)  # validators read the whole IR as it is now: their verdicts are about this version
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
        # the verdict is recorded as a tool-backed ``compile.<kind>`` result, so a refusal (FAIL) reaches
        # ``ir.validation``, RELEASE, the exit code and ``ai-eda review`` instead of living only in the stage table
        stamp = dict(check_id=f"compile.{kind}", tool=getattr(compiler, "id", type(compiler).__name__), tool_version=getattr(compiler, "version", None), ir_hash=ir.content_hash())
        try:
            ir.artifacts[kind] = compiler.compile(ir, CompileContext(workdir=ctx.workdir, tools=ctx.tools))
        except (NothingToCompileError, NotImplementedError) as e:
            ir.artifacts.pop(kind, None)  # an older artifact of this kind would be stale evidence
            ir.validation.add(ValidationResult(status=ValidationStatus.NOT_VERIFIED, message=str(e), **stamp))
            return ValidationStatus.NOT_VERIFIED, str(e)
        except CompileError as e:
            ir.artifacts.pop(kind, None)
            message = f"{kind} compile refused: {e}"
            ir.validation.add(ValidationResult(status=ValidationStatus.FAIL, message=message, details={"repair": "human"}, **stamp))
            return ValidationStatus.FAIL, message
        art = ir.artifacts[kind]
        # a cell the compiler neutralised (BOM free text written with a leading apostrophe) is reported, never hidden:
        # the full notes in the details, a count + the cells in the message
        details = {"neutralised": list(art.notes)} if art.notes else {}
        message = f"compiled {art.path}" + (f"; {_neutralised_summary(art.notes)}" if art.notes else "")
        ir.validation.add(ValidationResult(status=ValidationStatus.PASS, message=message, artifact_hash=art.content_hash, details=details, **stamp))
        return ValidationStatus.PASS, str(art.path)

    def _compile_stage(self, kind: ArtifactKind) -> StageFn:
        stage = Stage.SCHEMATIC if kind == ArtifactKind.SCHEMATIC else Stage.PCB

        def fn(ir: CircuitIR, ctx: AgentContext) -> StageOutcome:
            status, message = self._compile(ir, ctx, kind)
            if kind == ArtifactKind.SCHEMATIC:
                # the project file (fab limits as KiCad design rules) is written beside the schematic once there is
                # one, so ERC and DRC of this run read one project file; the schematic's message stays the stage
                # message verbatim, the project note is appended, its status folds in
                if status is ValidationStatus.PASS:
                    p_status, p_message = self._compile(ir, ctx, ArtifactKind.KICAD_PROJECT)
                    status = worst_status([status, p_status])
                    message = f"{message}; project file: {p_message}"
                else:
                    ir.artifacts.pop(ArtifactKind.KICAD_PROJECT, None)  # an older project file would be stale evidence
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
        notes: list[str] = []
        statuses: list[ValidationStatus] = []
        for kind in (ArtifactKind.BOM, ArtifactKind.CPL):
            # a BOM cell the compiler refuses (a formula-shaped identity) is a verdict on the IR, not a defect
            status, message = self._compile(ir, ctx, kind)
            art = ir.artifacts.get(kind)
            if status is not ValidationStatus.PASS:
                statuses.append(status)
                notes.append(f"{kind}: {message}")
            elif art is not None and art.notes:
                notes.append(f"{kind}: {_neutralised_summary(art.notes)}")
        notes.insert(0, "BOM/CPL compiled" if not statuses else "BOM/CPL: see below")
        if ArtifactKind.PCB not in ir.artifacts:
            notes.append("gerber/drill skipped (no PCB)")
            return StageOutcome(stage=stage, status=worst_status(statuses + [ValidationStatus.NOT_VERIFIED]), message="; ".join(notes))
        for kind in MANUFACTURING_EXPORTS:
            try:
                status, message = self._compile(ir, ctx, kind)
            except ToolUnavailableError as e:
                # the missing tool makes *this* export unverified; an earlier BOM/CPL FAIL in ``statuses`` still counts
                notes.append(f"{kind} export skipped: {e}")
                return StageOutcome(stage=stage, status=worst_status(statuses + [ValidationStatus.NOT_VERIFIED]), message="; ".join(notes))
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
        """RELEASE is PASS only on evidence: every latest result PASS (or NOT_APPLICABLE), tool-backed, and about this IR.

        A PASS without a ``tool`` is an opinion (ARCHITECTURE invariant 4); a
        PASS stamped with another IR version's hash is about a different
        design; a PASS without any stamp (an agent's result) is about this
        design only when this run produced it - one carried over from an
        earlier run in ``ir.validation`` vouches for nothing now. None of
        them releases anything, however the aggregate reads.
        """
        latest = ir.validation.latest_by_check()
        overall = ir.validation.overall()
        current = ir.content_hash()
        fresh = {r.check_id for r in ir.validation.results[getattr(self, "_fresh_from", 0):]}
        opinions = sorted(k for k, r in latest.items() if r.status == ValidationStatus.PASS and not r.is_tool_backed)
        stale = sorted(k for k, r in latest.items() if r.status == ValidationStatus.PASS and r.ir_hash and r.ir_hash != current)
        carried = sorted(k for k, r in latest.items() if r.status == ValidationStatus.PASS and not r.ir_hash and k not in fresh)
        if overall == ValidationStatus.PASS and not opinions and not stale and not carried:
            return StageOutcome(stage=Stage.RELEASE, status=ValidationStatus.PASS, message="evidence-backed release")
        reasons: list[str] = []
        if overall != ValidationStatus.PASS:
            blocking = sorted(k for k, r in latest.items() if r.status == overall)
            reasons.append(f"overall validation is {overall} ({', '.join(blocking)})")
        if opinions:
            reasons.append(f"PASS without a tool is an opinion, not evidence ({', '.join(opinions)})")
        if stale:
            reasons.append(f"PASS produced for another IR version ({', '.join(stale)})")
        if carried:
            reasons.append(f"PASS carried over from an earlier run without an IR version, not re-produced by this one ({', '.join(carried)})")
        status = overall if overall != ValidationStatus.PASS else ValidationStatus.NOT_VERIFIED
        return StageOutcome(stage=Stage.RELEASE, status=status, message="not releasable: " + "; ".join(reasons))
