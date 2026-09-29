"""The RF templates the selection considers after the five base templates.

Invariant: :data:`RF_TEMPLATES` holds only templates built from the RF design
library (:mod:`ai_eda.design.rf`) - each one a deterministic
:class:`~ai_eda.design.base.Template` like the base five, selected only by
confirmed requirements. ``design_from_requirements`` appends this list to
``TEMPLATES`` through a lazy import and tolerates it being empty: an empty
list selects nothing and changes no existing template's behaviour. It is
filled when the KR 447 MHz block templates are merged (wave 2 of the kr447
design, §5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_eda.design.base import Template

#: the RF templates, in stage order; empty until the block templates are merged
RF_TEMPLATES: list[Template] = []

__all__ = ["RF_TEMPLATES"]
