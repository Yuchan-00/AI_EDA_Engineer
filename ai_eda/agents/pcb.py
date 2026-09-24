"""PCB Agent: proposes a deterministic grid placement into ``ir.pcb``; DRC is done by KiCad, not here.

Invariants this agent keeps:

* It only *proposes*. It never mutates ``ir`` (the content hash before and
  after :meth:`PCBAgent.run` is the same); the orchestrator applies the
  proposal through ``Orchestrator.apply_proposals`` like any other.
* No LLM. The placement is :func:`ai_eda.tools.placement.grid.grid_placement`
  run on the footprints a :class:`~ai_eda.tools.kicad.library.KicadLibrary`
  (``ctx.tools["kicad_library"]``) found on disk; every placement carries
  ``derived`` / ``placement.grid`` provenance.
* It never guesses a footprint, never resizes a user outline and never
  places on top of existing copper: it proposes only when ``ir.pcb`` is
  ``None`` or has no placements, ``ir.pcb`` has no tracks / vias / zones,
  and every component has a footprint the library finds. Otherwise it
  proposes nothing and says why in a note starting ``not placed:``.
* It emits exactly one proposal, ``target="pcb"``, ``operation="set"``,
  whose payload is the whole :class:`~ai_eda.ir.PCBDesign` (the existing
  one copied with ``outline`` + ``placements`` filled, so layers and
  manufacturing constraints survive). It never emits a dotted ``pcb.*``
  target: ``apply_proposals`` resolves every target's parent *before* any
  step runs, so a second ``pcb.placements`` proposal in the same result would
  land on the old (replaced) object or raise on ``None``.
* Refusals the tool and the library report - ``CompileError`` (no
  components, no footprint, footprint not on disk, extent unknown, guard
  violated, user outline too small), ``LibraryLookupError`` and
  ``LibraryFormatError`` (a corrupt ``.kicad_mod``) - become a note and no
  proposal. Anything else propagates: it is a defect, not a verdict.
* It asks no question. The only steering is the control key
  :data:`PLACEMENT_KEY` (``--answer pcb.placement=skip`` proposes nothing;
  any other value is noted as not understood and placement proceeds); the
  key never becomes a requirement (``CONTROL_KEYS`` in
  :mod:`ai_eda.agents.requirement`).

The PLACEMENT stage runs before IR_BUILD, so this agent may see an
inconsistent IR (a net naming a component that does not exist). It walks
``ir.components`` only, so such an IR is still placed, and IR_BUILD then
reports the inconsistency exactly as before. The stage outcome is never
PASS: a proposal is not evidence; a placed-but-unrouted board FAILs DRC
with ``unconnected_items`` until someone routes it.
"""

from __future__ import annotations

from ai_eda.agents.base import Agent, AgentContext, AgentResult, IRProposal
from ai_eda.errors import CompileError
from ai_eda.ir import CircuitIR, PCBDesign
from ai_eda.llm.router import TaskKind
from ai_eda.tools.kicad.library import KicadLibrary, LibraryFormatError, LibraryLookupError
from ai_eda.tools.placement.grid import COLUMNS, MARGIN_MM, PLACER_ID, PLACER_VERSION, SPACING_MM, grid_placement

#: ``--answer pcb.placement=skip`` makes the agent propose nothing (a control key, never a requirement)
PLACEMENT_KEY = "pcb.placement"
SKIP_ANSWER = "skip"


class PCBAgent(Agent):
    name = "pcb"
    task = TaskKind.CIRCUIT_DESIGN

    def __init__(self, *, spacing_mm: float = SPACING_MM, margin_mm: float = MARGIN_MM, columns: int = COLUMNS) -> None:
        self.spacing_mm = spacing_mm
        self.margin_mm = margin_mm
        self.columns = columns

    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult:
        notes: list[str] = []
        answer = ctx.answers.get(PLACEMENT_KEY)
        if answer is not None:
            if answer.strip().lower() == SKIP_ANSWER:
                return self._result(notes=["placement skipped by answer"])
            notes.append(f"{PLACEMENT_KEY}={answer!r} not understood (the only answer is '{SKIP_ANSWER}'); placing as usual")
        reason = self._refusal(ir, ctx)
        if reason is not None:
            return self._result(notes=[*notes, f"not placed: {reason}"])
        library: KicadLibrary = ctx.tools["kicad_library"]
        base = ir.pcb if ir.pcb is not None else PCBDesign()
        try:
            grid = grid_placement(ir, library, spacing=self.spacing_mm, margin=self.margin_mm, columns=self.columns, outline=base.outline)
        except (CompileError, LibraryLookupError) as e:  # LibraryFormatError is a CompileError
            return self._result(notes=[*notes, f"not placed: {e}"])
        payload = base.model_copy(update={"outline": grid.outline, "placements": list(grid.placements)})
        o = grid.outline
        description = (
            f"{PLACER_ID} {PLACER_VERSION}: {len(grid.placements)} component(s) on a {o.width_mm} x {o.height_mm} mm "
            f"{'user' if base.outline is not None else 'generated'} outline at ({o.origin_x_mm}, {o.origin_y_mm}), "
            f"pitch {grid.pitch[0]} x {grid.pitch[1]} mm, {self.columns} column(s)"
        )
        proposal = IRProposal(
            description=description,
            target="pcb",
            operation="set",
            payload=payload,
            rationale="row-major grid from verified library footprint extents; placed extents are pairwise disjoint and inside the outline; DRC decides validity",
        )
        notes.append(f"{description}; unrouted: DRC will report unconnected_items until routed")
        return self._result(proposals=[proposal], notes=notes)

    @staticmethod
    def _refusal(ir: CircuitIR, ctx: AgentContext) -> str | None:
        """Why nothing is proposed, or ``None`` when the agent may place."""
        if ir.pcb is not None and ir.pcb.placements:
            return f"ir.pcb already has {len(ir.pcb.placements)} placement(s); the agent never replaces a layout"
        if ir.pcb is not None and (ir.pcb.tracks or ir.pcb.vias or ir.pcb.zones):
            return (
                f"ir.pcb has copper without placements ({len(ir.pcb.tracks)} track(s), {len(ir.pcb.vias)} via(s), "
                f"{len(ir.pcb.zones)} zone(s)); placing under existing copper would be a guess"
            )
        if not ir.components:
            return "no components to place"
        if not isinstance(ctx.tools.get("kicad_library"), KicadLibrary):
            return "no KiCad library in ctx.tools['kicad_library']; footprint extents cannot be read"
        return None


__all__ = ["PCBAgent", "PLACEMENT_KEY", "SKIP_ANSWER"]
