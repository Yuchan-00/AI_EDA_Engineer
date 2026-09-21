"""Validator framework.

A *validator* is a deterministic check that reads the IR (and optionally
derived artifacts) and returns :class:`ValidationResult` objects. Validators
are registered with the domains they apply to so that a power design gets
thermal analysis, an RF design gets impedance checks, and so on, without
every project paying for every check.
"""

from ai_eda.validation.base import ValidationContext, Validator
from ai_eda.validation.registry import ValidatorRegistry, default_registry

# Importing these modules registers the built-in validators on default_registry.
import ai_eda.validation.structural  # noqa: E402,F401
import ai_eda.validation.domain  # noqa: E402,F401

__all__ = ["ValidationContext", "Validator", "ValidatorRegistry", "default_registry"]
