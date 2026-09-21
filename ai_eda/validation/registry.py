from __future__ import annotations

from ai_eda.ir import CircuitIR, ValidationResult
from ai_eda.validation.base import ValidationContext, Validator


class ValidatorRegistry:
    def __init__(self) -> None:
        self._validators: dict[str, Validator] = {}

    def register(self, validator: Validator) -> Validator:
        if validator.id in self._validators:
            raise ValueError(f"validator id already registered: {validator.id}")
        self._validators[validator.id] = validator
        return validator

    def get(self, validator_id: str) -> Validator:
        return self._validators[validator_id]

    def all(self) -> list[Validator]:
        return list(self._validators.values())

    def select(self, ir: CircuitIR) -> list[Validator]:
        """Validators applicable to this design's domains."""
        return [v for v in self._validators.values() if v.applies_to(ir)]

    def run(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        for v in self.select(ir):
            results.extend(v.validate(ir, ctx))
        return results


default_registry = ValidatorRegistry()
