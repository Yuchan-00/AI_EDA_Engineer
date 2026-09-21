"""Manufacturing Agent: fab capability vs design, output-file checks.

Without authoritative fab data the verdict is NOT_VERIFIED - never "manufacturable".
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
        status = cap.verification_status()
        msg = "fab capability data is authoritative" if status == ValidationStatus.PASS else "fab capability limits not verified against official vendor data"
        return self._result(
            validation=[ValidationResult(check_id="mfg.capability", status=status, message=msg, details={"fab": cap.fab})]
        )
