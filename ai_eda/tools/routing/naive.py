"""NAIVE PLACEHOLDER ROUTER - NOT AN AUTOROUTER.

**Read this before using it.** :func:`route_naive` exists so the vertical
slice (IR -> ``.kicad_pcb`` -> real DRC -> gerbers) has *some* copper to
check. It is deliberately dumb and DRC-unaware:

* For every IR net it takes the absolute centres of the net's pads (from the
  verified library footprint + IR placement, via
  :mod:`ai_eda.tools.kicad.geometry`), sorts them by ``(x, y)`` and joins
  consecutive centres with **straight** segments on a single layer
  (``F.Cu`` by default).
* It knows nothing about clearance, crossings, courtyards, other nets, the
  board outline or which layer a pad is actually on. Two nets whose chains
  cross will short; a segment that grazes a foreign pad violates clearance;
  an SMD pad on the bottom side is simply not reachable from ``F.Cu``.
* Track width comes from the net's ``net_class`` through
  :data:`NET_CLASS_TRACK_WIDTH_MM` (``"Default"`` -> 0.25 mm); unknown
  classes fall back to :data:`DEFAULT_TRACK_WIDTH_MM`. Nothing here consults
  fab constraints.

The *only* statement about validity comes from ``kicad-cli pcb drc`` run by
:class:`ai_eda.tools.kicad.cli.KicadCli`. A layout that this router cannot
connect cleanly is a layout problem to be solved by a real router (future
work), never by loosening DRC.

Invariant kept: pad positions come from the library footprint file, never
from memory; a net pin whose component has no placement / footprint / pad
raises :class:`~ai_eda.errors.CompileError` instead of being skipped.

Traceability: every track returned carries ``derived`` provenance naming this
router (:data:`ROUTER_ID` / :data:`ROUTER_VERSION`), the net and the
placements it joined, so a segment on the board can be told apart from one a
human or a model drew.
"""

from __future__ import annotations

from ai_eda.errors import CompileError
from ai_eda.ir import CircuitIR, Net, Provenance, ProvenanceKind, Track
from ai_eda.tools.kicad.geometry import pad_center
from ai_eda.tools.kicad.library import KicadLibrary

__all__ = [
    "DEFAULT_LAYER",
    "DEFAULT_TRACK_WIDTH_MM",
    "NET_CLASS_TRACK_WIDTH_MM",
    "ROUTER_ID",
    "ROUTER_VERSION",
    "track_width_for",
    "net_pad_centers",
    "route_naive",
]

DEFAULT_LAYER = "F.Cu"
DEFAULT_TRACK_WIDTH_MM = 0.25
#: net class -> track width (mm). Placeholder table; real net classes belong in the IR constraints.
NET_CLASS_TRACK_WIDTH_MM: dict[str, float] = {"Default": 0.25}
#: provenance ``tool`` / ``tool_version`` stamped on every generated track
ROUTER_ID = "routing.naive"
ROUTER_VERSION = "0.1"


def track_provenance(net: Net) -> Provenance:
    """``derived`` provenance for a segment of ``net`` drawn by this router."""
    refs = sorted({p.component_ref for p in net.pins})
    return Provenance(
        kind=ProvenanceKind.DERIVED,
        tool=ROUTER_ID,
        tool_version=ROUTER_VERSION,
        derived_from=[f"net:{net.name}", *(f"placement:{r}" for r in refs)],
        note="straight pad-centre chain; validity is decided by kicad-cli DRC only",
    )


def track_width_for(net_class: str) -> float:
    """Track width for a net class; unknown classes get :data:`DEFAULT_TRACK_WIDTH_MM`."""
    return NET_CLASS_TRACK_WIDTH_MM.get(net_class, DEFAULT_TRACK_WIDTH_MM)


def net_pad_centers(ir: CircuitIR, net: Net, library: KicadLibrary) -> list[tuple[float, float]]:
    """Absolute centres of every pad on ``net`` (IR order). Refuses to guess.

    Raises :class:`CompileError` when ``ir.pcb`` is missing, a pin's component
    is unknown, unplaced, has no footprint, its footprint is not on disk, or
    the footprint has no pad with that pin number.
    """
    if ir.pcb is None:
        raise CompileError("cannot route: ir.pcb is None")
    centers: list[tuple[float, float]] = []
    for pin in net.pins:
        comp = ir.component(pin.component_ref)
        if comp is None:
            raise CompileError(f"net {net.name!r} references unknown component {pin.component_ref!r}")
        placement = ir.pcb.placement(comp.ref)
        if placement is None:
            raise CompileError(f"cannot route net {net.name!r}: component {comp.ref!r} has no placement")
        if comp.footprint is None:
            raise CompileError(f"cannot route net {net.name!r}: component {comp.ref!r} has no footprint")
        resolved = library.resolve_footprint(comp.footprint)
        if not resolved.verified:
            raise CompileError(
                f"cannot route net {net.name!r}: footprint {comp.footprint.library}:{comp.footprint.name} "
                f"of {comp.ref!r} was not found in a KiCad library"
            )
        pad = library.load_footprint(comp.footprint).pad(pin.pin_number)
        if pad is None:
            raise CompileError(
                f"net {net.name!r} references {comp.ref}.{pin.pin_number} but footprint "
                f"{comp.footprint.library}:{comp.footprint.name} has no pad {pin.pin_number!r}"
            )
        centers.append(pad_center(placement, pad))
    return centers


def route_naive(ir: CircuitIR, library: KicadLibrary | None = None, layer: str = DEFAULT_LAYER) -> list[Track]:
    """PLACEHOLDER: chain each net's pad centres with straight segments on ``layer``.

    Returns the tracks (it does not mutate ``ir``); the caller decides whether
    to store them in ``ir.pcb.tracks``. Nets are processed in name order and
    pads in ``(x, y)`` order so the result is deterministic. Coincident pad
    centres produce no zero-length segment. See the module docstring for
    everything this does *not* do.
    """
    lib = library or KicadLibrary()
    tracks: list[Track] = []
    for net in sorted(ir.nets, key=lambda n: n.name):
        points = sorted(set(net_pad_centers(ir, net, lib)))
        width = track_width_for(net.net_class)
        for start, end in zip(points, points[1:]):
            tracks.append(Track(net=net.name, layer=layer, start=start, end=end, width_mm=width, provenance=track_provenance(net)))
    return tracks
