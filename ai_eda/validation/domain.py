"""Domain-specific validators (scaffold).

Each returns NOT_VERIFIED until a real analysis backend exists. The point of
keeping them here now is that the *selection* mechanism is exercised end to
end and the GUI can show "this check exists but has not run".
"""

from __future__ import annotations

from ai_eda.ir import CircuitDomain, CircuitIR, ValidationResult
from ai_eda.validation.base import ValidationContext, Validator
from ai_eda.validation.registry import default_registry


class PowerThermalValidator(Validator):
    id = "domain.power.thermal"
    domains = frozenset({CircuitDomain.POWER})
    description = "Power dissipation and junction temperature vs datasheet limits"

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        return [self.not_verified("thermal analysis backend not implemented")]


class RFImpedanceValidator(Validator):
    id = "domain.rf.impedance"
    domains = frozenset({CircuitDomain.RF})
    description = "Trace impedance / matching / S-parameter checks"

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        return [self.not_verified("RF analysis backend not implemented")]


class SignalIntegrityValidator(Validator):
    id = "domain.high_speed.si"
    domains = frozenset({CircuitDomain.HIGH_SPEED})
    description = "Length matching, impedance control, crosstalk"

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        return [self.not_verified("signal integrity backend not implemented")]


class PowerIntegrityValidator(Validator):
    id = "domain.power.pi"
    domains = frozenset({CircuitDomain.POWER, CircuitDomain.HIGH_SPEED, CircuitDomain.DIGITAL})
    description = "Decoupling, PDN impedance, IR drop"

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        return [self.not_verified("power integrity backend not implemented")]


class AnalogBiasValidator(Validator):
    id = "domain.analog.bias"
    domains = frozenset({CircuitDomain.ANALOG, CircuitDomain.MIXED_SIGNAL})
    description = "Operating point, gain, noise (via SPICE)"

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        return [self.not_verified("analog analysis requires SPICE results; none attached")]


for _v in (
    PowerThermalValidator(),
    RFImpedanceValidator(),
    SignalIntegrityValidator(),
    PowerIntegrityValidator(),
    AnalogBiasValidator(),
):
    default_registry.register(_v)
