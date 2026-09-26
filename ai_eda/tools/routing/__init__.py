"""Routing tools.

:mod:`ai_eda.tools.routing.maze` is the pipeline's router: a deterministic
two-layer grid maze router (``F.Cu`` / ``B.Cu``, vias) whose every track and
via is ``derived`` from the IR placements, the KiCad footprints on disk and
its recorded parameters. :mod:`ai_eda.tools.routing.naive` is the old
straight-line **placeholder** kept as a fixture helper (it knows nothing
about clearances, crossings or layers). Whether a routed board is valid is
decided exclusively by real ``kicad-cli`` DRC
(:meth:`ai_eda.tools.kicad.cli.KicadCli.run_drc`), never by a router.
"""

from ai_eda.tools.routing.maze import (
    ROUTER_ID,
    ROUTER_VERSION,
    Routing,
    RoutingParams,
    effective_params,
    route_board,
)
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
    "ROUTER_ID",
    "ROUTER_VERSION",
    "Routing",
    "RoutingParams",
    "effective_params",
    "net_pad_centers",
    "route_board",
    "route_naive",
    "track_width_for",
]
