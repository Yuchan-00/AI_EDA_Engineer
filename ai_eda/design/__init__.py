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
    NO_RECORD,
    TEMPLATE_VERSION,
    TOOL_ID,
    UNVERIFIED_SUBSTITUTE,
    Choice,
    DesignChange,
    PartNote,
    Plan,
    Template,
    TheorySection,
    number,
    parameter_value,
    quantity,
    unserved_requirements,
    unverified,
)
from ai_eda.design.checks import INPUTS_CHECK, check_inputs_vs_requirements
from ai_eda.design.inputs import (
    DEFAULT_LAYER_COUNT,
    KEY_ALIASES,
    LAYER_COUNT_KEY,
    LAYER_COUNT_OPTIONS,
    PARSED_NOTE_PREFIX,
    UNIT_OF,
    DesignInput,
    LayerCountInput,
    canonical_key,
    is_template_input,
    read_inputs,
    read_layer_count,
    read_value,
)
from ai_eda.design.stackup import GENERIC_STACKS, board_layers, generic_stackup, plane_zones, stackup_choices, with_planes
from ai_eda.design.library_parts import TemplateRefusal, library_component, pin_by_name
from ai_eda.design.templates import TEMPLATES, Atmega128DevboardTemplate, AstableTemplate, DividerTemplate, LateLoad, LedTemplate, RcLowpassTemplate, design_from_requirements, late_load_changes, template_keys_text

__all__ = [
    "DEFAULT_LAYER_COUNT",
    "GENERIC_STACKS",
    "LAYER_COUNT_KEY",
    "LAYER_COUNT_OPTIONS",
    "LayerCountInput",
    "board_layers",
    "generic_stackup",
    "plane_zones",
    "read_layer_count",
    "stackup_choices",
    "with_planes",
    "CHOICE_NOTE_PREFIX",
    "CONFIRM_DESIGN_KEY",
    "DESIGN_CATEGORIES",
    "IGNORED_KEYS",
    "INPUTS_CHECK",
    "KEY_ALIASES",
    "LateLoad",
    "NO_RECORD",
    "PARSED_NOTE_PREFIX",
    "TEMPLATES",
    "TEMPLATE_VERSION",
    "TOOL_ID",
    "UNIT_OF",
    "UNVERIFIED_SUBSTITUTE",
    "Atmega128DevboardTemplate",
    "AstableTemplate",
    "Choice",
    "DesignChange",
    "DesignInput",
    "DividerTemplate",
    "LedTemplate",
    "PartNote",
    "Plan",
    "RcLowpassTemplate",
    "Template",
    "TemplateRefusal",
    "TheorySection",
    "canonical_key",
    "check_inputs_vs_requirements",
    "design_from_requirements",
    "is_template_input",
    "late_load_changes",
    "library_component",
    "number",
    "parameter_value",
    "pin_by_name",
    "quantity",
    "read_inputs",
    "read_value",
    "template_keys_text",
    "unserved_requirements",
    "unverified",
]
