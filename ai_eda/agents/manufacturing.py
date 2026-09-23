"""Manufacturing Agent: fab capability vs design, output-file checks.

Without authoritative fab data the verdict is NOT_VERIFIED - never
"manufacturable". Authoritative limits alone are not a verdict either: until a
check compares the board against them (none runs yet; the roadmap writes
them into the KiCad design rules for DRC), ``mfg.capability`` stays
NOT_VERIFIED and says which half is missing.
"""

from __future__ import annotations

from ai_eda.agents.base import Agent, AgentContext, AgentResult
from ai_eda.ir import CircuitIR, ValidationResult, ValidationStatus
from ai_eda.llm.router import TaskKind
from ai_eda.tools.manufacturing import FabCapability, jlcpcb_capability_unverified


class ManufacturingAgent(Agent):
    name = "manufacturing"
    task = TaskKind.RESULT_INTERPRETATION

    def run(self, ir: CircuitIR, ctx: AgentContext) -> AgentResult:
        cap: FabCapability = ctx.tools.get("fab_capability") or jlcpcb_capability_unverified()
        limits = cap.verification_status()
        # the limits' provenance is one question, whether this board was compared against them another: no
        # comparison runs yet, so the check never claims PASS (a PASS here would say "manufacturable")
        if limits == ValidationStatus.PASS:
            msg = "fab capability limits are authoritative, but the board is not compared against them (no capability check runs yet)"
        else:
            msg = "fab capability limits not verified against official vendor data"
        return self._result(
            validation=[ValidationResult(check_id="mfg.capability", status=ValidationStatus.NOT_VERIFIED, message=msg,
                                         details={"fab": cap.fab, "limits": limits, "compared": False})]
        )
