"""The RF design library of the KR 447 MHz licence-free FM walkie-talkie family.

Invariant: this package holds only deterministic, library-verified design
material - no model decides anything here. :mod:`~ai_eda.design.rf.parts`
is the parts table (library symbols / footprints, pins found by name,
compile rules checked first), :mod:`~ai_eda.design.rf.models` every
modelling number and SPICE card as a user-confirmed ``model.*`` choice
marked UNVERIFIED, :mod:`~ai_eda.design.rf.profile` the KR 447 MHz regulatory
placeholders (``kr447.*``, never grounded, never a PASS),
:mod:`~ai_eda.design.rf.family` the six ``radio_build`` boards with their
closed-world serve / need table and the selector question,
:mod:`~ai_eda.design.rf.blocks` the block builder API and
:mod:`~ai_eda.design.rf.registry` the RF templates the selection appends to
the base five (the four KR 447 MHz stage boards and the stage-5
transceiver with its conducted variant).
"""

from ai_eda.design.rf.family import BUILDS, SELECTOR_KEY, BuildInfo, selector_question, serving_builds, unserved_message
from ai_eda.design.rf.models import MODEL_PREFIX, MODEL_VALUES, MODEL_VERDICT, ModelCard, ModelValue, inductor_q_key, model_choice
from ai_eda.design.rf.parts import PARTS, REFUSED_PARTS, PartDef, PinSpec, PlacedPart, instantiate
from ai_eda.design.rf.profile import PROFILE, PROFILE_PREFIX, channel_plan, profile_choices, profile_keys, raster_refusal
from ai_eda.design.rf.registry import RF_TEMPLATES

__all__ = [
    "BUILDS",
    "MODEL_PREFIX",
    "MODEL_VALUES",
    "MODEL_VERDICT",
    "PARTS",
    "PROFILE",
    "PROFILE_PREFIX",
    "REFUSED_PARTS",
    "RF_TEMPLATES",
    "SELECTOR_KEY",
    "BuildInfo",
    "ModelCard",
    "ModelValue",
    "PartDef",
    "PinSpec",
    "PlacedPart",
    "channel_plan",
    "inductor_q_key",
    "instantiate",
    "model_choice",
    "profile_choices",
    "profile_keys",
    "raster_refusal",
    "selector_question",
    "serving_builds",
    "unserved_message",
]
