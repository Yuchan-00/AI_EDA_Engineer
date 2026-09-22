from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field

from ai_eda.ir import CircuitDomain, CircuitIR, ValidationResult, ValidationStatus


class ValidationContext(BaseModel):
    """Everything a validator may need besides the IR itself."""

    workdir: Path
    #: extra tool handles are injected by the orchestrator (kicad cli, spice runner, ...)
    tools: dict[str, object] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class Validator(ABC):
    """Base class for all validators.

    Subclasses set ``id`` and ``domains``. An empty ``domains`` set means the
    validator applies to every design (e.g. ERC, BOM consistency).
    """

    id: str = "validator.base"
    domains: frozenset[CircuitDomain] = frozenset()
    #: human description shown in the GUI
    description: str = ""
    #: check ids of tool results this validator reads (e.g. ``{"spice"}``); the orchestrator runs the
    #: validator again right after the stage that produced such a result, because at IR_BUILD time it
    #: can only say NOT_VERIFIED
    consumes: frozenset[str] = frozenset()

    def applies_to(self, ir: CircuitIR) -> bool:
        if not self.domains:
            return True
        if ir.topology is None:
            return False
        return bool(self.domains & set(ir.topology.domains))

    @abstractmethod
    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        """Run the check. Must never raise for 'could not check' - return NOT_VERIFIED instead."""

    # helpers ------------------------------------------------------------------

    def not_verified(self, message: str, **details) -> ValidationResult:
        return ValidationResult(check_id=self.id, status=ValidationStatus.NOT_VERIFIED, message=message, details=details)

    def not_applicable(self, message: str = "") -> ValidationResult:
        return ValidationResult(check_id=self.id, status=ValidationStatus.NOT_APPLICABLE, message=message)

    def user_input_required(self, message: str, **details) -> ValidationResult:
        return ValidationResult(
            check_id=self.id, status=ValidationStatus.USER_INPUT_REQUIRED, message=message, details=details
        )
