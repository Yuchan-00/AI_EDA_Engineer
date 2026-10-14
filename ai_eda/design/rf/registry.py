"""The RF templates the selection considers after the five base templates.

Invariant: :data:`RF_TEMPLATES` holds only templates built from the RF design
library (:mod:`ai_eda.design.rf`) - each one a deterministic
:class:`~ai_eda.design.base.Template` like the base five, selected only by
confirmed requirements. ``design_from_requirements`` appends this list to
``TEMPLATES`` through a lazy import and tolerates it being empty: an empty
list selects nothing and changes no existing template's behaviour.

The list holds the KR 447 MHz stage boards in the design's staging order
(kr447 design §2.0 / §5): ``kr447_audio_ptt`` (stage 1, ``radio_build =
audio_ptt``), ``kr447_rx_backend`` (stage 2, ``rx_backend``),
``kr447_rx_frontend`` (stage 3, ``rx_frontend``) and ``kr447_tx_exciter``
(stage 4, ``tx_exciter``), then the stage-5 ``kr447_transceiver``
(``radio_build = transceiver`` or ``transceiver_conducted``: one template,
two build variants). Each triggers only on its own confirmed
``radio_build`` value(s) (``Template.triggered_by``), so at most one of them
is ever selected and none triggers without ``radio_build``; the base five
never read ``radio_build``.
"""

from __future__ import annotations

from ai_eda.design.base import Template
from ai_eda.design.rf.t_audio_ptt import KR447_AUDIO_PTT
from ai_eda.design.rf.t_rx_backend import KR447RxBackendTemplate
from ai_eda.design.rf.t_rx_frontend import KR447RxFrontendTemplate
from ai_eda.design.rf.t_transceiver import KR447_TRANSCEIVER
from ai_eda.design.rf.t_tx_exciter import KR447_TX_EXCITER

#: the RF templates, in stage order (stage 1 .. 5)
RF_TEMPLATES: list[Template] = [
    KR447_AUDIO_PTT,
    KR447RxBackendTemplate(),
    KR447RxFrontendTemplate(),
    KR447_TX_EXCITER,
    KR447_TRANSCEIVER,
]

__all__ = ["RF_TEMPLATES"]
