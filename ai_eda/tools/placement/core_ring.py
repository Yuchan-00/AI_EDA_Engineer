"""Deterministic core-and-ring placement for a board built around one many-pad part (pure, no I/O beyond the library).

Invariant: every coordinate produced here is a function of the IR's
components and nets, the footprints read from a KiCad library
(:class:`~ai_eda.tools.kicad.library.KicadLibrary`: extents and pad
positions - never model memory) and the parameters :data:`SPACING_MM` /
:data:`MARGIN_MM`. Nothing is estimated and nothing is guessed: a component
without a footprint, a footprint that is not on disk or one whose extent
cannot be measured raises :class:`~ai_eda.errors.CompileError` (the same
refusals as :func:`ai_eda.tools.placement.grid.grid_placement`), and so does a
ring that cannot hold its parts - there is no fallback to overlapping parts.

What this is: the placement the PCB agent uses when a part has at least
:data:`CORE_MIN_PADS` pads (a 64-pin TQFP microcontroller, say):

1. **Core.** The part with the most pads (distinct pad numbers; ties go to
   the natural ref order, ``U1 < U2``) sits at the centre of the board,
   rotation 0, top side; its extent box (courtyard union pads) is centred on
   the board centre.
2. **Pull angle.** Every other part gets the circular mean of the directions,
   seen from the core centre, of the core pads it connects to (angles
   counter-clockwise on screen, 0 = east). Rails - nets of kind ``POWER`` /
   ``GROUND`` - count only for a part that reaches the core through no other
   net (a crystal capacitor follows its XTAL pin, not the many ground pads;
   a decoupling capacitor follows the supply pads). Parts joined by nets that
   reach no core pad form a *group* (a header and its series resistors, an
   LED and its resistor, the regulator's input parts); a part without a
   signal path of its own to the core takes its group's angle - the mean
   over the core pads the whole group reaches through signal nets, else
   through rails - so the group stays together (the digest's "one hop" to
   the part it connects to, extended to the whole group). A part with no
   angle at all (no net, or directions that cancel out) is packed after the
   others on the outer ring in natural ref order. Parts whose angles are
   equal (every member of a group, and a rail-only part whose rails give the
   same mean) are ordered by where the angle came from (so a group is never
   split by another part of the same angle), then along the group's own
   chain - a breadth-first walk over the nets that reach no core pad,
   starting at the group's first connector / switch (else its first member)
   and taking neighbours in natural ref order, so a DC jack, its diode, the
   input capacitor and the regulator follow each other in that order - then
   by natural ref.
3. **Rings.** The *inner* ring holds the parts with at most
   :data:`INNER_MAX_PADS` pads whose every connected pin's net also reaches
   a core pad (decoupling, crystal and its capacitors, reset R / C,
   pull-ups) and that are not connectors or switches (reference prefix in
   :data:`EDGE_REF_PREFIXES`); its band starts :data:`SPACING_MM` outside
   the core extent and each part touches the band's inner edge. Every other
   part is on the *outer* ring, whose band ends :data:`MARGIN_MM` inside the
   board edge (each part touches the band's outer edge, so connectors sit at
   the edge). Each band is four straight runs, one per side, walked
   counter-clockwise from the south-east corner (east side going north,
   north going west, west going south, south going east); a run owns its
   outer corner and starts :data:`SPACING_MM` past the neighbouring band at
   its inner corner, so parts on different runs never touch. Parts are
   sorted by the position their pull angle points at on that walk; each is
   put at that position when it is free, else just after its predecessor
   (never straddling a corner), and a chain that overruns the walk is pushed
   back from its end. A part lies tangential - rotated 0 / 90 / 180 / 270
   so that its longer extent runs along the side (a header's pin row
   parallel to the edge) - and of the two tangential rotations it takes the
   one whose pad 1 is further out for a connector / switch (the pin-1 row
   outward on a 2-row header), then the one whose pads lie closer to their
   nets' other ends - the core pads of a net that reaches the core, else the
   centres of the other parts on the net (so an LED and its resistor, or a
   diode and the capacitor after it, turn their shared pads toward each
   other) -, then the smaller angle. The exception is a connector / switch
   with a *body*: one whose extent (courtyard union pads, both from the
   library) reaches at least :data:`BODY_OVERHANG_MM` further past its pads
   on one side than on the opposite side - a barrel jack's body, a
   receptacle's shell. It is turned radially, the one rotation that puts
   that side on the run's outward normal, so the body overhangs toward the
   board edge (its extent's far end :data:`MARGIN_MM` inside the outline)
   instead of pointing along the edge at its neighbour; its extent across
   the run then sets the outer band's depth. Which end of a body is the
   opening is not read from anything (the footprint does not say); the rule
   only guarantees that the body's long side ends at the edge with no part
   in front of it.
4. **Outline.** Without a user outline the outer band grows in
   :data:`GROW_STEP_MM` steps from its smallest size (``SPACING_MM`` outside
   the inner band) until its parts fit; the outline is the outermost band
   plus :data:`MARGIN_MM`, its half sizes rounded up to
   :data:`OUTLINE_QUANTUM_MM`, at origin ``(0, 0)`` with the core at its
   centre (on a whole millimetre, so on the 0.2 and 0.25 mm routing grids).
   A user outline (``ir.pcb.outline``) is kept verbatim - never resized -
   with the core at its centre and the outer band at its edge; parts that do
   not fit are refused. The inner band's size is set by the core, so an
   inner ring that cannot hold its parts is refused either way.

Before returning, the placed extents are re-measured with
:func:`~ai_eda.tools.kicad.geometry.footprint_bbox`: any two that touch or
overlap, or a part outside the outline, is a :class:`CompileError` - the
guard behind the arithmetic, as in the grid placer.

What this is not: a layout judgement. Whether the board is valid is decided
only by ``kicad-cli pcb drc`` on the compiled board; signal integrity,
thermal relief and the distance from a decoupling capacitor to its supply
pin are not considered, and a connector's mating direction is considered
only through the body rule of step 3.

Traceability: every :class:`~ai_eda.ir.Placement` carries ``derived``
provenance naming this tool (:data:`PLACER_ID` / :data:`PLACER_VERSION`) with
the footprint, the ring, the pull angle and where it came from, the core and
the parameters in ``derived_from``. ``Provenance.inputs`` stays empty: that
field is the calculator role map, and a placement is not a calculator output.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.errors import CompileError
from ai_eda.ir import BoardOutline, BoardSide, CircuitIR, NetKind, Placement, Provenance, ProvenanceKind
from ai_eda.tools.kicad.geometry import footprint_bbox, pad_center, pads_bbox
from ai_eda.tools.kicad.library import BBox, FootprintDef, KicadLibrary
from ai_eda.tools.placement.grid import MARGIN_MM, SPACING_MM, _disjoint, _inside, _q, _resolve, footprint_extent

__all__ = [
    "PLACER_ID",
    "PLACER_VERSION",
    "CORE_MIN_PADS",
    "INNER_MAX_PADS",
    "EDGE_REF_PREFIXES",
    "BODY_OVERHANG_MM",
    "GROW_STEP_MM",
    "OUTLINE_QUANTUM_MM",
    "SPACING_MM",
    "MARGIN_MM",
    "RingPlacement",
    "pad_count",
    "body_axis",
    "find_core",
    "core_ring_placement",
    "ring_provenance",
]

#: provenance ``tool`` / ``tool_version`` stamped on every placement
PLACER_ID = "placement.core_ring"
PLACER_VERSION = "0.2"
#: a part with at least this many pads makes the PCB agent use this placer instead of the grid
CORE_MIN_PADS = 32
#: parts with more pads never go on the inner ring
INNER_MAX_PADS = 4
#: reference prefixes of the parts that belong at the board edge (connector / plug, switch), never on the inner ring
EDGE_REF_PREFIXES: tuple[str, ...] = ("J", "P", "SW")
#: a connector / switch whose extent reaches this much further past its pads on one side than on the opposite one has a body
#: (a barrel jack's body) and is turned so that side faces out of the board (module docstring, step 3)
BODY_OVERHANG_MM = 3.0
#: how much the generated outer band grows on every side per attempt when its parts do not fit
GROW_STEP_MM = 1.0
#: the generated outline's half sizes are rounded up to this, so the core centre lands on a whole millimetre
OUTLINE_QUANTUM_MM = 1.0
#: net kinds that count as rails for the pull angle (module docstring, step 2)
RAIL_KINDS = frozenset({NetKind.POWER, NetKind.GROUND})

_EPS = 1e-9
_PREFIX = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True, slots=True)
class RingPlacement:
    """What :func:`core_ring_placement` produced (board frame, mm)."""

    outline: BoardOutline
    placements: list[Placement]
    extents: dict[str, BBox]
    core: str
    core_pads: int
    #: ref -> "core" | "inner" | "outer"
    rings: dict[str, str]
    #: ref -> pull angle in degrees (counter-clockwise on screen, 0 = east), ``None`` without one
    angles: dict[str, float | None]
    #: ref -> where the pull angle came from: "signal", "rail", "group:<ref>+<ref>..." or "none"
    pulls: dict[str, str]
    #: the inner edge of each band that exists: {"inner": box, "outer": box}
    bands: dict[str, BBox] = field(default_factory=dict)
    #: ref -> (run, start): the run (0 east, 1 north, 2 west, 3 south) and the mm along its ring's walk where the part begins
    walk: dict[str, tuple[int, float]] = field(default_factory=dict)


def pad_count(fp: FootprintDef) -> int:
    """Distinct non-empty pad numbers: the pads a net can reach (two pads sharing a number are one logical pad)."""
    return len({p.number for p in fp.pads if p.number})


def _edge_part(ref: str) -> bool:
    m = _PREFIX.match(ref)
    return m is not None and m.group(0).upper() in EDGE_REF_PREFIXES


def body_axis(fp: FootprintDef) -> str | None:
    """``"x"`` / ``"y"``: the axis (footprint frame, rotation 0) along which the extent overhangs the pads by at least
    :data:`BODY_OVERHANG_MM` more on one side than on the other (the larger difference wins, x on a tie); ``None`` without such a body."""
    ext = footprint_extent(fp)
    pads = pads_bbox(Placement(component_ref="", x_mm=0.0, y_mm=0.0, rotation_deg=0.0, side=BoardSide.TOP), fp)
    if pads is None:
        return None
    dx = abs((pads.x1 - ext.x1) - (ext.x2 - pads.x2))
    dy = abs((pads.y1 - ext.y1) - (ext.y2 - pads.y2))
    if max(dx, dy) < BODY_OVERHANG_MM - _EPS:
        return None
    return "x" if dx >= dy else "y"


def find_core(parts: list[tuple[str, FootprintDef]]) -> tuple[str, FootprintDef, int]:
    """The part with the most pads; ``parts`` in natural ref order, so the first of equals wins."""
    if not parts:
        raise CompileError("no components to place")
    best = parts[0]
    best_n = pad_count(best[1])
    for ref, fp in parts[1:]:
        n = pad_count(fp)
        if n > best_n:
            best, best_n = (ref, fp), n
    return best[0], best[1], best_n


def ring_provenance(footprint_id: str, ring: str, angle: float | None, pull: str, core: str, spacing: float, margin: float) -> Provenance:
    """``derived`` provenance of a placement this tool made (the ring, the pull angle and its source, the parameters)."""
    return Provenance(
        kind=ProvenanceKind.DERIVED,
        tool=PLACER_ID,
        tool_version=PLACER_VERSION,
        derived_from=[
            f"footprint:{footprint_id}",
            f"ring:{ring}",
            f"pull_angle_deg:{'none' if angle is None else round(angle, 3)}",
            f"pull:{pull}",
            f"core:{core}",
            f"spacing_mm:{spacing}",
            f"margin_mm:{margin}",
        ],
        note="core in the centre, inner ring by pull angle, outer ring at the edge, from library footprint extents and the IR nets; "
        "validity is decided by kicad-cli DRC only",
    )


# --------------------------------------------------------------------------- geometry


def _box_at(fp: FootprintDef, rotation: float) -> BBox:
    """The footprint's extent around its own origin at ``rotation`` on the top side (exact at multiples of 90 degrees)."""
    box = footprint_bbox(Placement(component_ref="", x_mm=0.0, y_mm=0.0, rotation_deg=rotation, side=BoardSide.TOP), fp)
    assert box is not None  # footprint_extent measured it at rotation 0 already (it raises when it cannot)
    return box


def _expand(b: BBox, d: float) -> BBox:
    return BBox(b.x1 - d, b.y1 - d, b.x2 + d, b.y2 + d)


@dataclass(slots=True)
class _Part:
    ref: str
    fp: FootprintDef
    length: float  # extent along the side
    depth: float  # extent across it
    angle: float | None
    pull: str
    edge: bool
    body: bool = False  # a connector / switch with a body: turned radially, the body toward the edge (module docstring, step 3)
    chain: int = 0  # position along its group's chain (module docstring, step 2); 0 outside a group
    target: float | None = None  # walk position the angle points at (the part's centre)
    run: int = -1
    start: float = 0.0


class _Band:
    """One ring: the four runs around the rectangle ``r`` (its inner edge), ``depth`` deep, parts touching the ``align`` edge."""

    def __init__(self, r: BBox, depth: float, spacing: float, align: str) -> None:
        self.r, self.d, self.sp, self.align = r, depth, spacing, align
        lengths = (
            (r.y2 - spacing) - (r.y1 - depth),  # east, going north from the south-east inner corner
            (r.x2 - spacing) - (r.x1 - depth),  # north, going west
            (r.y2 + depth) - (r.y1 + spacing),  # west, going south
            (r.x2 + depth) - (r.x1 + spacing),  # south, going east
        )
        self.lengths = [max(0.0, v) for v in lengths]
        self.starts = [sum(self.lengths[:k]) for k in range(4)]
        self.total = sum(self.lengths)

    def target(self, angle: float) -> float:
        """Walk position of the point where the ray at ``angle`` meets the band's centre line (clamped into its run)."""
        r, d, sp = self.r, self.d, self.sp
        a = math.radians(angle)
        dx, dy = math.cos(a), -math.sin(a)  # screen frame, y down
        hx, hy = r.x2 + d / 2.0, r.y2 + d / 2.0  # the band is centred on the core centre (0, 0)
        tx = hx / abs(dx) if abs(dx) > _EPS else math.inf
        ty = hy / abs(dy) if abs(dy) > _EPS else math.inf
        t = min(tx, ty)
        x, y = t * dx, t * dy
        if tx <= ty:
            k, off = (0, (r.y2 - sp) - y) if dx > 0 else (2, y - (r.y1 + sp))
        else:
            k, off = (1, (r.x2 - sp) - x) if dy < 0 else (3, x - (r.x1 + sp))
        return self.starts[k] + min(max(off, 0.0), self.lengths[k])

    def run_of(self, u: float, *, end: bool = False) -> int:
        """Index of the run holding walk position ``u`` (a boundary belongs to the later run, or to the earlier one when ``end``)."""
        for k in range(4):
            lo, hi = self.starts[k], self.starts[k] + self.lengths[k]
            if (lo < u <= hi + _EPS) if end else (lo - _EPS <= u < hi):
                return k
        return 4  # past the walk

    def box(self, run: int, start: float, length: float, depth: float) -> BBox:
        """The board-frame box a part ``length`` x ``depth`` occupies at walk position ``start`` on ``run``."""
        r, d, sp = self.r, self.d, self.sp
        off = start - self.starts[run]
        inner = self.align == "inner"
        if run == 0:
            y2 = (r.y2 - sp) - off
            x1 = r.x2 if inner else r.x2 + d - depth
            return BBox(x1, y2 - length, x1 + depth, y2)
        if run == 1:
            x2 = (r.x2 - sp) - off
            y1 = r.y1 - depth if inner else r.y1 - d
            return BBox(x2 - length, y1, x2, y1 + depth)
        if run == 2:
            y1 = (r.y1 + sp) + off
            x1 = r.x1 - depth if inner else r.x1 - d
            return BBox(x1, y1, x1 + depth, y1 + length)
        x1 = (r.x1 + sp) + off
        y1 = r.y2 if inner else r.y2 + d - depth
        return BBox(x1, y1, x1 + length, y1 + depth)


def _pack(band: _Band, parts: list[_Part], gap: float) -> bool:
    """Place ``parts`` (already in walk order) on the band's runs; ``False`` when they do not fit."""
    if not parts:
        return True
    # forward: each part at its target (its centre) when free, else just after its predecessor; a part never straddles a corner;
    # past the last run the walk continues virtually (run 4) so that the backward pass below knows how far the chain overran
    cursor = 0.0
    for p in parts:
        s = cursor if p.target is None else max(cursor, p.target - p.length / 2.0)
        s = max(s, 0.0)
        k = band.run_of(s)
        while k < 4 and s + p.length > band.starts[k] + band.lengths[k] + _EPS:
            k += 1
            if k < 4:
                s = max(s, band.starts[k])
        p.run, p.start = k, s
        cursor = s + p.length + gap
    if all(p.run < 4 for p in parts):
        return True
    # backward: push the chain back from the end of the walk, each part as little as needed
    limit = band.total
    for p in reversed(parts):
        e = min(limit, p.start + p.length)
        while True:
            k = band.run_of(e, end=True)
            if k >= 4 or e <= 0.0:
                return False
            if e - p.length >= band.starts[k] - _EPS:
                break
            e = band.starts[k]  # would straddle the corner: end at the previous run's end instead
        p.run, p.start = k, e - p.length
        limit = p.start - gap
    return all(0 <= p.run < 4 and p.start >= -_EPS for p in parts)


# --------------------------------------------------------------------------- pull angles


def _circular_mean(points: list[tuple[float, float]]) -> float | None:
    """Mean direction (degrees, counter-clockwise on screen, 0 = east) of ``points`` seen from (0, 0); ``None`` when they cancel out."""
    sx = sum(math.cos(math.atan2(-y, x)) for x, y in points if (x, y) != (0.0, 0.0))
    sy = sum(math.sin(math.atan2(-y, x)) for x, y in points if (x, y) != (0.0, 0.0))
    if math.hypot(sx, sy) <= 1e-9 * max(1, len(points)):
        return None
    return round(math.degrees(math.atan2(sy, sx)) % 360.0, 9) % 360.0


def _pull_angles(
    ir: CircuitIR, parts: list[tuple[str, FootprintDef]], core: str, core_pads: dict[str, list[tuple[float, float]]]
) -> tuple[dict[str, tuple[float | None, str]], dict[str, int]]:
    """``(ref -> (angle, source), ref -> chain position)`` for every part but the core (module docstring, step 2).

    ``source`` is ``"signal"`` / ``"rail"`` (the part's own core pads),
    ``"group:<refs>"`` (its group's, the members joined by ``+`` in natural
    order) or ``"none"``. The chain position is the part's place in the
    breadth-first walk of its group over the nets that reach no core pad,
    from the group's first connector / switch (else its first member),
    neighbours in natural ref order; 0 for a part alone.
    """
    refs = [ref for ref, _fp in parts if ref != core]
    nets_of: dict[str, list[str]] = {ref: [] for ref in refs}
    kinds = {n.name: n.kind for n in ir.nets}
    parent = {ref: ref for ref in refs}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for net in ir.nets:
        members: list[str] = []
        for pin in net.pins:
            ref = pin.component_ref
            if ref in nets_of and ref not in members:
                members.append(ref)
                if net.name not in nets_of[ref]:
                    nets_of[ref].append(net.name)
        if net.name not in core_pads:  # a net that reaches no core pad joins its parts into one group
            for other in members[1:]:
                a, b = sorted((find(members[0]), find(other)), key=natural_ref_key)
                parent[b] = a
    groups: dict[str, list[str]] = {}
    for ref in sorted(refs, key=natural_ref_key):
        groups.setdefault(find(ref), []).append(ref)

    def pads(members: list[str], rail: bool) -> list[tuple[float, float]]:
        return [pt for ref in members for name in nets_of[ref] if (kinds[name] in RAIL_KINDS) == rail for pt in core_pads.get(name, [])]

    out: dict[str, tuple[float | None, str]] = {}
    for ref in refs:
        group = groups[find(ref)]
        candidates: list[tuple[list[tuple[float, float]], str]] = [(pads([ref], False), "signal")]
        if len(group) > 1:
            label = "group:" + "+".join(group)
            candidates += [(pads(group, False), label), (pads(group, True), label)]
        candidates.append((pads([ref], True), "rail"))
        out[ref] = (None, "none")
        for points, source in candidates:
            angle = _circular_mean(points) if points else None
            if angle is not None:
                out[ref] = (angle, source)
                break
    chain: dict[str, int] = {}
    own = {ref: {name for name in nets_of[ref] if name not in core_pads} for ref in refs}
    for members in groups.values():
        start = next((ref for ref in members if _edge_part(ref)), members[0])
        order, seen, k = [start], {start}, 0
        while k < len(order):
            cur = order[k]
            k += 1
            for other in members:
                if other not in seen and own[cur] & own[other]:
                    seen.add(other)
                    order.append(other)
        order += [ref for ref in members if ref not in seen]
        chain.update({ref: i for i, ref in enumerate(order)})
    return out, chain


# --------------------------------------------------------------------------- orientation


def _rotations(part: _Part) -> tuple[float, ...]:
    """The two tangential rotations (the longer extent along the run; a square part keeps 0 / 180), or for a part with a body
    the one rotation whose extent overhangs its pads furthest along the run's outward normal (the smaller angle on a tie)."""
    fp, run = part.fp, part.run
    if part.body:
        best: tuple[float, float] | None = None
        for rot in (0.0, 90.0, 180.0, 270.0):
            e = _box_at(fp, rot)
            p = pads_bbox(Placement(component_ref="", x_mm=0.0, y_mm=0.0, rotation_deg=rot, side=BoardSide.TOP), fp)
            assert p is not None  # body_axis measured the pads already
            out = (e.x2 - p.x2, p.y1 - e.y1, p.x1 - e.x1, e.y2 - p.y2)[run]
            if best is None or out > best[0] + _EPS:
                best = (out, rot)
        assert best is not None
        return (best[1],)
    b = footprint_extent(fp)
    horizontal_run = run in (1, 3)
    wide = b.width > b.height + _EPS
    tall = b.height > b.width + _EPS
    if (horizontal_run and tall) or (not horizontal_run and wide):
        return 90.0, 270.0
    return 0.0, 180.0


def _outward(run: int, x: float, y: float) -> float:
    """How far ``(x, y)`` lies along the run's outward normal (east +x, north -y, west -x, south +y)."""
    return (x, -y, -x, y)[run]


def _orient(
    part: _Part,
    box: BBox,
    core_pads: dict[str, list[tuple[float, float]]],
    pin_net: dict[tuple[str, str], str],
    partners: dict[str, list[tuple[str, tuple[float, float]]]],
) -> Placement:
    """The placement of ``part`` whose rotated extent fills ``box`` (local frame), choosing between its rotations (:func:`_rotations`).

    The cost of a rotation sums, over the part's pads, the distance to the
    nearest other end of the pad's net: a core pad when the net reaches the
    core, else the centre of another part's box on the net (``partners``).
    """
    best: tuple | None = None
    for rot in _rotations(part):
        ext = _box_at(part.fp, rot)
        pl = Placement(component_ref=part.ref, x_mm=box.x1 - ext.x1, y_mm=box.y1 - ext.y1, rotation_deg=rot, side=BoardSide.TOP)
        cost = 0.0
        pin1: float | None = None
        for pad in part.fp.pads:
            cx, cy = pad_center(pl, pad)
            if pad.number == "1" and pin1 is None:
                pin1 = _outward(part.run, cx, cy)
            name = pin_net.get((part.ref, pad.number)) if pad.number else None
            if name is None:
                continue
            targets = core_pads.get(name) or [centre for ref, centre in partners.get(name, []) if ref != part.ref]
            if targets:
                cost += min(math.hypot(cx - tx, cy - ty) for tx, ty in targets)
        key = (-round(pin1, 6) if part.edge and pin1 is not None else 0.0, round(cost, 6), rot)
        if best is None or key < best[0]:
            best = (key, pl)
    assert best is not None
    return best[1]


# --------------------------------------------------------------------------- the placer


def core_ring_placement(
    ir: CircuitIR,
    library: KicadLibrary,
    *,
    spacing: float = SPACING_MM,
    margin: float = MARGIN_MM,
    outline: BoardOutline | None = None,
) -> RingPlacement:
    """Place every component of ``ir`` around its many-pad core; pure (same IR + library -> same result).

    Raises :class:`CompileError` for: no components, a component without a
    footprint, a footprint not on disk, an unmeasurable extent, a parameter
    that makes no sense, a ring that cannot hold its parts (the inner ring
    always; the outer ring only inside a user outline, which is never
    resized) and a result that violates the guard.
    """
    if not (spacing > 0) or not (margin >= 0):
        raise CompileError("spacing must be > 0 and margin >= 0 (parts on neighbouring runs are kept apart by the spacing)")
    parts = _resolve(ir, library)
    core, core_fp, core_n = find_core(parts)
    ext = footprint_extent(core_fp)
    # local frame: (0, 0) is the centre of the core's extent box
    core_local = Placement(component_ref=core, x_mm=-(ext.x1 + ext.x2) / 2.0, y_mm=-(ext.y1 + ext.y2) / 2.0, rotation_deg=0.0, side=BoardSide.TOP)
    core_box = footprint_bbox(core_local, core_fp)
    assert core_box is not None
    pin_net = {(pin.component_ref, pin.pin_number): net.name for net in ir.nets for pin in net.pins}
    core_pads: dict[str, list[tuple[float, float]]] = {}
    for pad in core_fp.pads:
        name = pin_net.get((core, pad.number)) if pad.number else None
        if name is not None:
            core_pads.setdefault(name, []).append(pad_center(core_local, pad))
    pulls, chain = _pull_angles(ir, parts, core, core_pads)
    inner: list[_Part] = []
    outer: list[_Part] = []
    for ref, fp in parts:
        if ref == core:
            continue
        box = footprint_extent(fp)
        angle, pull = pulls[ref]
        axis = body_axis(fp) if _edge_part(ref) else None
        if axis is None:
            length, depth = max(box.width, box.height), min(box.width, box.height)
        else:  # radial: the body's axis runs across the side
            length, depth = (box.height, box.width) if axis == "x" else (box.width, box.height)
        part = _Part(ref, fp, length, depth, angle, pull, _edge_part(ref), body=axis is not None, chain=chain[ref] if pull.startswith("group:") else 0)
        connected = {pin_net[(ref, pad.number)] for pad in fp.pads if pad.number and (ref, pad.number) in pin_net}
        to_core = bool(connected) and all(name in core_pads for name in connected)
        (inner if to_core and pad_count(fp) <= INNER_MAX_PADS and not part.edge else outer).append(part)

    r_in = _expand(core_box, spacing)
    d_in = max((p.depth for p in inner), default=0.0)
    band_in = _Band(r_in, d_in, spacing, "inner")
    _order(band_in, inner)
    if not _pack(band_in, inner, spacing):
        raise CompileError(
            f"the inner ring cannot hold its {len(inner)} part(s) ({', '.join(p.ref for p in inner)}): they need "
            f"{_q(sum(p.length for p in inner) + spacing * (len(inner) - 1))} mm along a band of {_q(band_in.total)} mm "
            f"({spacing} mm around the {_q(core_box.width)} x {_q(core_box.height)} mm core {core}, {_q(d_in)} mm deep); "
            f"refusing rather than overlapping parts"
        )

    d_out = max((p.depth for p in outer), default=0.0)
    r_out_min = _expand(r_in, d_in + spacing) if inner else r_in
    if outline is not None:
        centre = (outline.origin_x_mm + outline.width_mm / 2.0, outline.origin_y_mm + outline.height_mm / 2.0)
        hx, hy = outline.width_mm / 2.0 - margin - d_out, outline.height_mm / 2.0 - margin - d_out
        if outer and (hx < r_out_min.x2 - _EPS or hy < r_out_min.y2 - _EPS):
            raise CompileError(
                f"the {outline.width_mm} x {outline.height_mm} mm outline at ({outline.origin_x_mm}, {outline.origin_y_mm}) leaves no room "
                f"for the outer ring ({_q(d_out)} mm deep, {margin} mm from the edge) around the core and the inner ring; "
                f"the outline is kept as given, not resized"
            )
        band_out = _Band(BBox(-hx, -hy, hx, hy), d_out, spacing, "outer")
        _order(band_out, outer)
        if not _pack(band_out, outer, spacing):
            raise CompileError(
                f"the outer ring cannot hold its {len(outer)} part(s) inside the {outline.width_mm} x {outline.height_mm} mm outline: they need "
                f"{_q(sum(p.length for p in outer) + spacing * (len(outer) - 1))} mm along a band of {_q(band_out.total)} mm; "
                f"the outline is kept as given, not resized"
            )
    else:
        # the outline's half sizes are rounded up to the quantum first and the outer band is then made flush with it
        # (margin inside), so the connectors sit exactly at the edge and the rounding widens the channel between the rings
        grow = 0.0
        limit = sum(p.length + spacing for p in outer) + GROW_STEP_MM
        while True:
            far = _expand(r_out_min, grow + d_out) if outer else (_expand(r_in, d_in) if inner else core_box)
            hx = _ceil(max(-far.x1, far.x2) + margin)
            hy = _ceil(max(-far.y1, far.y2) + margin)
            band_out = _Band(BBox(-hx + margin + d_out, -hy + margin + d_out, hx - margin - d_out, hy - margin - d_out), d_out, spacing, "outer")
            _order(band_out, outer)
            if _pack(band_out, outer, spacing):
                break
            grow += GROW_STEP_MM
            if grow > limit:  # every run is longer than all the parts together by now: unreachable, kept as a guard
                raise CompileError(f"the outer ring cannot hold its {len(outer)} part(s) at any size; refusing")
        outline = BoardOutline(width_mm=_q(2 * hx), height_mm=_q(2 * hy), origin_x_mm=0.0, origin_y_mm=0.0)
        centre = (hx, hy)

    rings: dict[str, str] = {core: "core"}
    local: list[tuple[Placement, FootprintDef, str]] = [(core_local, core_fp, "core")]
    boxes = {p.ref: band.box(p.run, p.start, p.length, p.depth) for band, members in ((band_in, inner), (band_out, outer)) for p in members}
    partners: dict[str, list[tuple[str, tuple[float, float]]]] = {}
    for net in ir.nets:
        if net.name in core_pads:
            continue
        for pin in net.pins:
            b = boxes.get(pin.component_ref)
            if b is not None and all(ref != pin.component_ref for ref, _c in partners.get(net.name, [])):
                partners.setdefault(net.name, []).append((pin.component_ref, ((b.x1 + b.x2) / 2.0, (b.y1 + b.y2) / 2.0)))
    for band, members, ring in ((band_in, inner, "inner"), (band_out, outer, "outer")):
        for p in members:
            local.append((_orient(p, boxes[p.ref], core_pads, pin_net, partners), p.fp, ring))
            rings[p.ref] = ring
    cx, cy = centre
    placements: list[Placement] = []
    placed: dict[str, BBox] = {}
    for pl, fp, ring in sorted(local, key=lambda t: natural_ref_key(t[0].component_ref)):
        ref = pl.component_ref
        angle, pull = pulls.get(ref, (None, "core"))
        final = Placement(
            component_ref=ref,
            x_mm=_q(cx + pl.x_mm),
            y_mm=_q(cy + pl.y_mm),
            rotation_deg=pl.rotation_deg,
            side=BoardSide.TOP,
            provenance=ring_provenance(fp.lib_id, ring, angle, pull, core, spacing, margin),
        )
        box = footprint_bbox(final, fp)
        assert box is not None
        placements.append(final)
        placed[ref] = box
    refs = [p.component_ref for p in placements]
    for a_i, a in enumerate(refs):
        for b in refs[a_i + 1:]:
            if not _disjoint(placed[a], placed[b]):
                raise CompileError(f"placed extents of {a!r} and {b!r} touch or overlap ({placed[a]} vs {placed[b]}); refusing the placement")
    outside = [r for r in refs if not _inside(placed[r], outline)]
    if outside:
        raise CompileError(
            f"component(s) {outside} do not fit inside the {outline.width_mm} x {outline.height_mm} mm outline at "
            f"({outline.origin_x_mm}, {outline.origin_y_mm}) with margin {margin} mm; the outline is kept as given, not resized"
        )
    bands = {"inner": _shift(band_in.r, cx, cy)} if inner else {}
    if outer:
        bands["outer"] = _shift(band_out.r, cx, cy)
    return RingPlacement(
        outline=outline,
        placements=placements,
        extents=placed,
        core=core,
        core_pads=core_n,
        rings=rings,
        angles={ref: pulls[ref][0] for ref in refs if ref != core},
        pulls={ref: pulls[ref][1] for ref in refs if ref != core},
        bands=bands,
        walk={p.ref: (p.run, _q(p.start)) for p in (*inner, *outer)},
    )


def _order(band: _Band, parts: list[_Part]) -> None:
    """Sort ``parts`` in walk order: by the position their angle points at, then angle, then where the angle came from (a group
    stays together), then the group chain, then natural ref; parts without an angle last (module docstring, step 2)."""
    for p in parts:
        p.target = None if p.angle is None else band.target(p.angle)
    parts.sort(key=lambda p: (
        p.target is None, p.target if p.target is not None else 0.0, p.angle if p.angle is not None else 0.0, p.pull, p.chain, natural_ref_key(p.ref),
    ))


def _ceil(v: float) -> float:
    return math.ceil(round(v / OUTLINE_QUANTUM_MM, 9)) * OUTLINE_QUANTUM_MM


def _shift(b: BBox, dx: float, dy: float) -> BBox:
    return BBox(_q(b.x1 + dx), _q(b.y1 + dy), _q(b.x2 + dx), _q(b.y2 + dy))
