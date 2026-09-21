"""Routing tools.

Only :mod:`ai_eda.tools.routing.naive` exists today and it is a **placeholder**:
it draws straight copper between pad centres with no knowledge of clearances,
crossings or layers. Whether a routed board is valid is decided exclusively by
real ``kicad-cli`` DRC (:meth:`ai_eda.tools.kicad.cli.KicadCli.run_drc`), never
by the router.
"""

from ai_eda.tools.routing.naive import (
    DEFAULT_LAYER,
    DEFAULT_TRACK_WIDTH_MM,
    NET_CLASS_TRACK_WIDTH_MM,
    net_pad_centers,
    route_naive,
    track_width_for,
)

__all__ = [
    "DEFAULT_LAYER",
    "DEFAULT_TRACK_WIDTH_MM",
    "NET_CLASS_TRACK_WIDTH_MM",
    "net_pad_centers",
    "route_naive",
    "track_width_for",
]
