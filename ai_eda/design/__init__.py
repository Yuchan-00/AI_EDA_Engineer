"""Deterministic circuit templates: the only way a circuit enters an empty IR without a human writing it.

A template is selected by confirmed requirement values, computes every
number with a registered calculator, instantiates parts from the KiCad
library on disk and presents its free choices under the required question
``confirm_design``; the circuit agent applies the plan only once the user
confirmed that table (:mod:`ai_eda.design.base` states the invariant). No
model is involved anywhere in this package.
"""

from ai_eda.design.base import (
    CHOICE_NOTE_PREFIX,
    CONFIRM_DESIGN_KEY,
    DESIGN_CATEGORIES,
    IGNORED_KEYS,
    TEMPLATE_VERSION,
    TOOL_ID,
    Choice,
    DesignChange,
    Plan,
    Template,
    unserved_requirements,
)
from ai_eda.design.checks import INPUTS_CHECK, check_inputs_vs_requirements
from ai_eda.design.inputs import KEY_ALIASES, PARSED_NOTE_PREFIX, UNIT_OF, DesignInput, canonical_key, is_template_input, read_inputs, read_value
from ai_eda.design.library_parts import TemplateRefusal, library_component, pin_by_name
from ai_eda.design.templates import TEMPLATES, AstableTemplate, DividerTemplate, LateLoad, LedTemplate, RcLowpassTemplate, design_from_requirements, late_load_changes, template_keys_text

__all__ = [
    "CHOICE_NOTE_PREFIX",
    "CONFIRM_DESIGN_KEY",
    "DESIGN_CATEGORIES",
    "IGNORED_KEYS",
    "INPUTS_CHECK",
    "KEY_ALIASES",
    "LateLoad",
    "PARSED_NOTE_PREFIX",
    "TEMPLATES",
    "TEMPLATE_VERSION",
    "TOOL_ID",
    "UNIT_OF",
    "AstableTemplate",
    "Choice",
    "DesignChange",
    "DesignInput",
    "DividerTemplate",
    "LedTemplate",
    "Plan",
    "RcLowpassTemplate",
    "Template",
    "TemplateRefusal",
    "canonical_key",
    "check_inputs_vs_requirements",
    "design_from_requirements",
    "is_template_input",
    "late_load_changes",
    "library_component",
    "pin_by_name",
    "read_inputs",
    "read_value",
    "template_keys_text",
    "unserved_requirements",
]
