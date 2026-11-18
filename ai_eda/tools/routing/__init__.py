"""Routing tools.

:mod:`ai_eda.tools.routing.maze` is the pipeline's router: a deterministic
two-layer grid maze router (``F.Cu`` / ``B.Cu``, vias) with negotiated
congestion (rip-up and reroute until no net's copper lies in another's
clearance halo) whose every track and via is ``derived`` from the IR
placements, the KiCad footprints on disk and its recorded parameters; it
emits whole nets only; given a :class:`RouterStack` (:mod:`ai_eda.tools.routing.stack`)
it routes with the stack's via types and stamps the board
:data:`ROUTER_STACK_VERSION` - on all of the stack's signal layers (N layers:
a barrel-state search with through, blind, buried and micro signal vias,
VPR's negotiation schedule and a rip-up-and-reroute end game) when the stack
has inner signal layers or a second signal via type; with ``corner_style="miter45"``
(:mod:`ai_eda.tools.routing.corners`, routing.corners 0.1 - the project owner's
corner-style rule, a geometry convention, not an electrical requirement) its
right-angle corners are mitred into 45-degree chamfers last, under the same
exact clearance audit; with ``corner_style="arc"`` (routing.corners 0.2, the
PCB agent's default, the same convention) its sidesteps - short tracks
between two right-angle corners - are straightened first and its corners
then rounded into tangent arcs (tracks with ``mid``), mitred where no arc
fits; the router's own default ``"orthogonal"`` keeps its grid corners
byte for byte. :mod:`ai_eda.tools.routing.naive` is the old
straight-line **placeholder** kept as a fixture helper (it knows nothing
about clearances, crossings or layers). Whether a routed board is valid is
decided exclusively by real ``kicad-cli`` DRC
(:meth:`ai_eda.tools.kicad.cli.KicadCli.run_drc`), never by a router.
"""

from ai_eda.tools.routing.maze import (
    FINE_PITCH_MM,
    FINE_RULES,
    ROUTER_ID,
    ROUTER_STACK_VERSION,
    ROUTER_VERSION,
    Routing,
    RoutingParams,
    effective_params,
    finest_pad_pitch,
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
from ai_eda.tools.routing.stack import RouterStack, ViaType

__all__ = [
    "DEFAULT_LAYER",
    "DEFAULT_TRACK_WIDTH_MM",
    "FINE_PITCH_MM",
    "FINE_RULES",
    "NET_CLASS_TRACK_WIDTH_MM",
    "ROUTER_ID",
    "ROUTER_STACK_VERSION",
    "ROUTER_VERSION",
    "RouterStack",
    "Routing",
    "RoutingParams",
    "ViaType",
    "effective_params",
    "finest_pad_pitch",
    "net_pad_centers",
    "route_board",
    "route_naive",
    "track_width_for",
]
