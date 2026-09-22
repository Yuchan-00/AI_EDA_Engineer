"""Simulation Agent: compiles the SPICE netlist from the IR, runs every analysis of the
simulation setup on a real engine and judges the expectations - nothing else.

The agent never decides what to simulate or what the answer should be: both
live in ``ir.simulation`` (traced, and refused by the netlist compiler when
``llm_generated``). It returns no proposals; its output is evidence
(:class:`~ai_eda.ir.ValidationResult` ``spice`` + ``spice.<expectation>``,
each with the netlist hash, the engine version and the rawfiles) produced by
:func:`ai_eda.tools.spice.stage.run_spice_for`, the same function the repair
loop's ``RerunTool`` uses.

Verdicts when it cannot simulate: no simulation setup, no engine, no compiler
or nothing bound yet -> NOT_VERIFIED; an IR the netlist compiler refuses
(``CompileError``: an unbound part, an ``llm_generated`` value, a floating
pin ...) -> FAIL with the compiler's reason and ``repair: human``. In every
such case the ``spice.<id>`` results of an earlier run are superseded
(:func:`~ai_eda.tools.spice.stage.retire_expectation_results`): a verdict
about a netlist that no longer exists must not stay the latest word.
"""

from __future__ import annotations

from ai_eda.agents.base import Agent, AgentContext, AgentResult
from ai_eda.compilers.base import CompileContext
from ai_eda.errors import CompileError, NothingToCompileError, ToolUnavailableError
from ai_eda.ir import ArtifactKind, CircuitIR, ValidationResult, ValidationStatus
from ai_eda.llm.router import TaskKind
from ai_eda.tools.spice import SpiceRunner
from ai_eda.tools.spice.stage import CHECK_ID, retire_expectation_results, run_spice_for


class SimulationAgent(Agent):
    name = "simulation"
    task = TaskKind.RESULT_INTERPRETATION

    def _verdict(self, *results: ValidationResult) -> AgentResult:
        """The results plus the summary's message as the stage note (the orchestrator shows notes, not results)."""
        return self._result(validation=list(results), notes=[results[0].message])

    def _superseding(self, ir: CircuitIR, summary: ValidationResult) -> AgentResult:
        """``summary`` plus a NOT_VERIFIED successor for every ``spice.<id>`` this run did not produce."""
        retired = retire_expectation_results(ir, set(), ValidationStatus.NOT_VERIFIED, f"no result for the current IR: {summary.message}", ir_hash=ir.content_hash())
        return self._verdict(summary, *retired)

    def _not_verified(self, ir: CircuitIR, message: str) -> AgentResult:
        return self._superseding(ir, ValidationResult(check_id=CHECK_ID, status=ValidationStatus.NOT_VERIFIED, message=message))

    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult:
        if ir.simulation is None:
            return self._not_verified(ir, "no simulation setup in IR")
        runner: SpiceRunner | None = ctx.tools.get("spice")
        if runner is None or not runner.available():
            return self._not_verified(ir, "no SPICE engine available")
        compiler = ctx.tools.get("compilers", {}).get(ArtifactKind.SPICE_NETLIST)
        if compiler is None:
            return self._not_verified(ir, "no SPICE netlist compiler registered")
        try:
            ir.artifacts[ArtifactKind.SPICE_NETLIST] = compiler.compile(ir, CompileContext(workdir=ctx.workdir, tools=ctx.tools))
        except NothingToCompileError as e:
            self._drop_stale(ir)
            return self._not_verified(ir, str(e))
        except CompileError as e:
            self._drop_stale(ir)
            return self._superseding(
                ir,
                ValidationResult(
                    check_id=CHECK_ID,
                    status=ValidationStatus.FAIL,
                    message=f"spice netlist compile refused: {e}",
                    tool=compiler.id,
                    tool_version=compiler.version,
                    ir_hash=ir.content_hash(),
                    details={"repair": "human", "reason": str(e)},
                ),
            )
        try:
            results = run_spice_for(ir, ctx.tools, ctx.workdir)
        except ToolUnavailableError as e:
            ir.artifacts.pop(ArtifactKind.SPICE_RESULT, None)
            return self._not_verified(ir, f"SPICE engine unavailable: {e}")
        return self._verdict(*results)

    @staticmethod
    def _drop_stale(ir: CircuitIR) -> None:
        """A refused compile must not leave an older netlist / result set behind as evidence."""
        ir.artifacts.pop(ArtifactKind.SPICE_NETLIST, None)
        ir.artifacts.pop(ArtifactKind.SPICE_RESULT, None)
