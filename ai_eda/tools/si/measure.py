"""What the routed copper of each net is, electrically: length, via barrels, propagation delay, impedance per segment.

Invariant: every number here is computed from the IR copper (``ir.pcb.tracks``
/ ``ir.pcb.vias``), the board's stackup (``ir.pcb.stackup``) and the registered
transmission-line calculators (:mod:`ai_eda.tools.calc.tline`) - nothing is
estimated from a picture and nothing is guessed. What cannot be computed is
said, never replaced by a number:

* no stackup -> no delay and no impedance (:data:`~ai_eda.tools.calc.tline.NO_STACKUP`);
* a track on a layer whose adjacent copper is not a plane has no impedance
  (:data:`~ai_eda.tools.calc.tline.NO_REFERENCE_PLANE`). Its delay is
  bounded, not computed: a line's field lies partly in the dielectrics and
  partly in air, so its effective permittivity is at most the largest er
  its field can reach - with no plane to stop it, every dielectric of the
  stack (on a 2-layer board the one dielectric; on a stack whose core er
  exceeds the prepreg's, the core's) - and its delay per length at most
  ``calc.tline.tpd(er_max)`` = sqrt(er_max)/c0 (the via barrel's bound too).
  The bound is conservative for the critical-length rule (it makes a line
  look longer, never shorter) and every result that uses it says so
  (``bound``); the ids name the dielectric whose er it is;
* over a plane the segment is a microstrip: Z0 and e_eff from
  ``calc.tline.microstrip.*`` (Hammerstad & Jensen) at the routed width, the
  delay per length ``calc.tline.tpd(e_eff)``.

* over a plane *layer* whose zone a keep-out forbids under the track (a
  ``PCBDesign.keepouts`` area that forbids ``zones`` on the reference layer
  and does not allow the plane's net, :func:`plane_keepout`; a polygon
  keep-out by its bounding box, exactly as the PCB agent clips the plane
  zones, so the two never disagree - conservative) there is no
  reference plane either: "over a plane" is decided per layer by the
  stackup, and a keep-out that clears the plane from under a feed line must
  not leave it judged as a microstrip. Such a track is counted apart in its
  segment (``keepout_mm`` / ``keepout_reason`` / ``keepout_id``): it has no
  impedance (the impedance check makes it NOT_VERIFIED with that reason) and
  its delay is the no-plane upper bound, like a track without a plane - the
  largest er of the stack (``keepout_bound`` names it: with the plane
  cleared the field reaches the core toward the next plane). Every check
  that reports such a net as having no reference plane names the keep-out
  (:func:`no_plane_reason`) - "use pcb_layers=4 or add a plane" is the
  remedy only for a layer the stack gives no plane. Where the copper merely
  comes near the plane's edge the fringing field is not judged. A board
  without keep-outs is measured exactly as before.

A via is a through via (the compiler writes it so): its barrel spans the
stack from ``F.Cu`` to ``B.Cu`` (:meth:`~ai_eda.ir.Stackup.span_mm`) and its
delay per length is bounded by the largest er of the dielectrics it crosses
(the same bound as above: its field is in the dielectric). A net's routed
length is its tracks plus one barrel per via - the length the router's
``max_length_mm`` budget counts - and its delay is the sum over its segments
and vias. Both are totals of the net's whole copper, an upper bound of any
pad-to-pad path through it: summed branches are not a line.

The *line* of a net (:attr:`NetMeasure.line`) is what the critical-length
rule and ``spice.si`` measure: the longest pad-to-pad path through the copper
(:func:`ai_eda.tools.si.paths.longest_path`, extracted when the caller gives
the KiCad library the pad boxes come from), else - for a net with exactly
two pads - its whole copper (the line between its two pads). A net with more
pads whose path could not be extracted has no line: only the total's upper
bound is known (``path_problem`` says why).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ai_eda.ir import CircuitIR, Stackup
from ai_eda.tools.keepout import allowed_nets, area_bbox, area_points, covers_layer, forbids, keepout_id, keepouts_of, segment_area_distance
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.si.paths import NetPath, longest_path, net_pads
from ai_eda.tools.calc.tline import (
    NO_REFERENCE_PLANE,
    NO_STACKUP,
    TLineRangeError,
    line_geometry,
    microstrip,
    propagation_delay,
)

#: tool id / version of the SI measurements and checks (IR geometry + calculators, never DRC)
SI_TOOL = "si"
SI_VERSION = "0.1"
#: what every SI result says it is
NOT_DRC = "IR geometry + calculators, not DRC; the fab's measured impedance is the only real one"
#: how the reason of a track without a reference plane because of a keep-out begins (:func:`plane_keepout`)
KEEPOUT_NO_PLANE = f"{NO_REFERENCE_PLANE} under keep-out"


def _zone_keepouts(ir: CircuitIR) -> list:
    """The board's keep-outs that forbid zones (empty for a board without keep-outs)."""
    return [k for k in keepouts_of(ir.pcb) if forbids(k, "zones")]


def _track_plane_keepout(stackup: Stackup | None, track, keepouts: list) -> tuple[str, str] | None:
    """``(keep-out id, reason)`` of the first zone keep-out that clears the track's plane from under its copper, or ``None``."""
    if not keepouts or stackup is None:
        return None
    geom, _why = line_geometry(stackup, track.layer)
    if geom is None:
        return None  # no plane in the stack under this layer anyway
    layer, net = geom.reference_layer, str(geom.reference_net)
    a = (float(track.start[0]), float(track.start[1]))
    b = (float(track.end[0]), float(track.end[1]))
    half = float(track.width_mm) / 2.0
    for k in keepouts:
        if not covers_layer(k, layer) or net in allowed_nets(k):
            continue
        x1, y1, x2, y2 = area_bbox(area_points(k))  # by its box, as the PCB agent clips the planes: the two never disagree
        if segment_area_distance(a, b, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]) < half - 1e-9:
            reason = str(getattr(k, "reason", "") or "").strip()
            return keepout_id(k), (f"{KEEPOUT_NO_PLANE} {keepout_id(k)}: it forbids zones on {layer} (the {net} plane) under this copper"
                                   + (f" ({reason})" if reason else ""))
    return None


def plane_keepout(ir: CircuitIR, track) -> str | None:
    """Why ``track`` has no reference plane because a keep-out forbids its plane's zone under its copper, or ``None`` (module docstring).

    The plane is the stackup's reference layer of the track's layer
    (:func:`~ai_eda.tools.calc.tline.line_geometry`); a keep-out counts when
    it covers that layer, forbids ``zones``, does not allow the plane's net
    and the track's copper (centreline within half its width) reaches its
    area (a polygon by its bounding box, as the plane clipping cuts it).
    ``None`` for a board without such a keep-out or a track without a
    plane layer at all (that reason is the stackup's own).
    """
    pcb = ir.pcb
    if pcb is None:
        return None
    hit = _track_plane_keepout(pcb.stackup, track, _zone_keepouts(ir))
    return None if hit is None else hit[1]


@dataclass(frozen=True)
class LineModel:
    """A track of ``width_mm`` on ``layer``: its Z0 (``None`` with ``reason`` when undefined) and its delay per length.

    ``bound`` is True when ``t_pd`` is the upper bound sqrt(er)/c0 (no
    reference plane, or a geometry outside the microstrip formula's range),
    False when it is the microstrip's own sqrt(e_eff)/c0. ``ids`` are the
    stackup ids of the inputs (h, t, er) the calculators read.
    """

    layer: str
    width_mm: float
    z0_ohm: float | None
    e_eff: float
    t_pd_s_per_m: float
    bound: bool
    reason: str | None
    reference: str | None = None
    ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def _outer_dielectric(stackup: Stackup, layer: str) -> int | None:
    i = stackup.index(layer)
    if i == 0:
        return 0
    if i == len(stackup.copper) - 1:
        return len(stackup.dielectrics) - 1
    return None


def line_model(stackup: Stackup | None, layer: str, width_mm: float) -> tuple[LineModel | None, str | None]:
    """The electrical model of a track (module docstring), or ``(None, reason)`` when not even a delay bound exists."""
    if stackup is None:
        return None, NO_STACKUP
    if layer not in stackup.copper_names():
        return None, f"layer {layer!r} is not in the stackup {stackup.copper_names()}"
    geom, why = line_geometry(stackup, layer)
    if geom is not None:
        try:
            r = microstrip(width_mm, float(geom.h.value), float(geom.t.value), float(geom.er.value))
        except (TLineRangeError, ValueError) as e:
            why = f"microstrip formula refuses {width_mm:g} mm on {layer}: {e}"
        else:
            return LineModel(
                layer=layer, width_mm=width_mm, z0_ohm=r.z0_ohm, e_eff=r.e_eff, t_pd_s_per_m=propagation_delay(r.e_eff), bound=False, reason=None,
                reference=f"{geom.reference_layer} ({geom.reference_net})", ids=geom.ids, notes=geom.notes,
            ), None
    d = _outer_dielectric(stackup, layer)
    if d is None:
        return None, why or f"{layer} is an inner layer"
    er, k = _bound_er(stackup, d)
    return LineModel(
        layer=layer, width_mm=width_mm, z0_ohm=None, e_eff=er, t_pd_s_per_m=propagation_delay(er), bound=True, reason=why,
        ids=(f"pcb.stackup.dielectrics[{k}].er",),
    ), None


def _bound_er(stackup: Stackup, outer: int) -> tuple[float, int]:
    """``(er, index)`` of the no-plane delay bound: the largest er of the stack (module docstring), the layer's own dielectric on a tie."""
    ers = [float(d.er.value) for d in stackup.dielectrics]
    top = max(ers)
    return top, outer if ers[outer] == top else ers.index(top)


def via_model(stackup: Stackup | None) -> tuple[float, float] | None:
    """``(barrel length mm, delay per length s/m)`` of a through via, or ``None`` without a stackup."""
    if stackup is None:
        return None
    er_max = max(float(d.er.value) for d in stackup.dielectrics)
    return stackup.span_mm("F.Cu", "B.Cu"), propagation_delay(er_max)


@dataclass
class Segment:
    """All of a net's copper on one layer at one width: total length and its line model.

    ``keepout_mm`` of that length lies where a keep-out clears the plane from
    under the copper (module docstring): no impedance there, and its delay
    is the no-plane upper bound ``keepout_t_pd`` (``keepout_reason`` says
    which keep-out).
    """

    layer: str
    width_mm: float
    length_mm: float
    model: LineModel | None
    reason: str | None = None
    keepout_mm: float = 0.0
    keepout_reason: str | None = None
    keepout_t_pd: float | None = None
    #: the id of the keep-out that clears the plane, and which dielectric's er the bound ``keepout_t_pd`` is (``pcb.stackup.dielectrics[k].er``)
    keepout_id: str | None = None
    keepout_bound: str | None = None

    @property
    def delay_s(self) -> float | None:
        if self.model is None:
            return None
        if self.keepout_mm > 0.0 and self.keepout_t_pd is not None:
            over = max(self.length_mm - self.keepout_mm, 0.0)
            return over / 1000.0 * self.model.t_pd_s_per_m + self.keepout_mm / 1000.0 * self.keepout_t_pd
        return self.length_mm / 1000.0 * self.model.t_pd_s_per_m


@dataclass(frozen=True)
class NetLine:
    """The line a net's critical length and SPICE deck use: its length and delay, how it was measured, and its main width."""

    length_mm: float
    delay_s: float | None
    bound: bool
    #: "path" (the longest pad-to-pad path) or "two-pad copper" (the whole copper of a 2-pad net)
    how: str
    ends: tuple[str, str] | None
    #: ``(layer, width_mm)`` with the most track length along the line
    main: tuple[str, float] | None

    @property
    def t_pd_s_per_m(self) -> float | None:
        return None if self.delay_s is None or self.length_mm <= 0 else self.delay_s / (self.length_mm / 1000.0)

    def describe(self) -> str:
        if self.how == "path" and self.ends is not None:
            return f"the longest pad-to-pad path {self.ends[0]}-{self.ends[1]}"
        return "the copper of the 2-pad net"


@dataclass
class NetMeasure:
    """One net's routed copper, electrically (module docstring). ``delay_s`` is ``None`` with ``problems`` when it cannot be computed."""

    net: str
    track_length_mm: float = 0.0
    vias: int = 0
    via_length_mm: float = 0.0
    via_delay_s: float = 0.0
    segments: list[Segment] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    #: the net's logical pads in the IR (distinct ``(ref, pin)``)
    pads: int = 0
    #: the longest pad-to-pad path (``None``: not extracted, ``path_problem`` says why)
    path: NetPath | None = None
    path_problem: str | None = None

    @property
    def routed(self) -> bool:
        return self.track_length_mm > 0 or self.vias > 0

    @property
    def length_mm(self) -> float:
        """Tracks plus one barrel per via (the router's ``max_length_mm`` measure)."""
        return self.track_length_mm + self.vias * self.via_length_mm

    @property
    def delay_s(self) -> float | None:
        if self.problems:
            return None
        return sum(s.delay_s or 0.0 for s in self.segments) + self.vias * self.via_delay_s

    @property
    def bound(self) -> bool:
        """Whether any part of the delay is the no-plane upper bound (a segment without a plane, or copper under a keep-out)."""
        return any(s.model is not None and (s.model.bound or s.keepout_mm > 0.0) for s in self.segments)

    @property
    def keepout_reasons(self) -> list[str]:
        """The distinct reasons (segment order) a keep-out clears this net's reference plane from under some of its copper."""
        return list(dict.fromkeys(s.keepout_reason for s in self.segments if s.keepout_mm > 0.0 and s.keepout_reason))

    @property
    def keepout_ids(self) -> list[str]:
        """The keep-outs (ids, segment order) that clear this net's reference plane from under some of its copper."""
        return list(dict.fromkeys(s.keepout_id for s in self.segments if s.keepout_mm > 0.0 and s.keepout_id))

    @property
    def plane_less(self) -> bool:
        """Whether some of the net's copper is on a layer the stack gives no reference plane (the no-plane bound for that reason)."""
        return any(s.model is not None and s.model.bound for s in self.segments)

    @property
    def t_pd_s_per_m(self) -> float | None:
        """The net's effective delay per length (delay / routed length)."""
        d = self.delay_s
        return None if d is None or self.length_mm <= 0 else d / (self.length_mm / 1000.0)

    @property
    def main_segment(self) -> Segment | None:
        """The segment (layer, width) with the most copper."""
        return max(self.segments, key=lambda s: (s.length_mm, -s.width_mm), default=None)

    @property
    def line(self) -> NetLine | None:
        """The net's line (module docstring): the extracted path, else the whole copper of a 2-pad net, else ``None``."""
        if not self.routed:
            return None
        if self.path is not None:
            p = self.path
            return NetLine(p.length_mm, p.delay_s if not self.problems else None, p.bound, "path", p.ends, p.main)
        if self.pads == 2:
            main = self.main_segment
            return NetLine(self.length_mm, self.delay_s, self.bound, "two-pad copper", None, None if main is None else (main.layer, main.width_mm))
        return None

    def summary(self) -> dict:
        return {
            "net": self.net, "length_mm": _r(self.length_mm), "track_length_mm": _r(self.track_length_mm), "vias": self.vias,
            "via_length_mm": _r(self.via_length_mm), "delay_s": None if self.delay_s is None else float(f"{self.delay_s:.6e}"), "bound": self.bound,
            "segments": [
                {"layer": s.layer, "width_mm": s.width_mm, "length_mm": _r(s.length_mm),
                 "z0_ohm": None if s.model is None or s.model.z0_ohm is None else round(s.model.z0_ohm, 4),
                 "t_pd_ps_per_mm": None if s.model is None else round(s.model.t_pd_s_per_m * 1e9, 6), "bound": None if s.model is None else s.model.bound,
                 "reason": s.reason or (None if s.model is None else s.model.reason),
                 **({"keepout_mm": _r(s.keepout_mm), "keepout_reason": s.keepout_reason, "keepout_bound": s.keepout_bound} if s.keepout_mm > 0.0 else {})}
                for s in self.segments
            ],
            "problems": list(self.problems),
        }


def _r(x: float) -> float:
    return round(x, 6)


def measure_nets(ir: CircuitIR, nets: list[str] | None = None, library: KicadLibrary | None = None) -> dict[str, NetMeasure]:
    """:class:`NetMeasure` of every IR net (or of ``nets``), in IR net order; pure (the IR is only read).

    With ``library`` (the KiCad footprints on disk) each routed net's longest
    pad-to-pad path is extracted (:mod:`ai_eda.tools.si.paths`); without it
    ``path_problem`` says so.
    """
    pcb = ir.pcb
    names = [n.name for n in ir.nets] if nets is None else list(nets)
    out = {name: NetMeasure(net=name) for name in names}
    for n in ir.nets:
        if n.name in out:
            out[n.name].pads = len({(p.component_ref, p.pin_number) for p in n.pins})
    if pcb is None:
        return out
    stackup = pcb.stackup
    per: dict[tuple[str, str, float], float] = {}
    kos = _zone_keepouts(ir)
    under: dict[tuple[str, str, float], list] = {}  # key -> [length under a zone-forbidding keep-out, first reason]
    for t in pcb.tracks:
        m = out.get(t.net)
        if m is None:
            continue
        length = math.hypot(t.end[0] - t.start[0], t.end[1] - t.start[1])
        m.track_length_mm += length
        key = (t.net, t.layer, float(t.width_mm))
        per[key] = per.get(key, 0.0) + length
        if kos:
            hit = _track_plane_keepout(stackup, t, kos)
            if hit is not None:
                row = under.setdefault(key, [0.0, hit[1], hit[0]])
                row[0] += length
    vm = via_model(stackup)
    for v in pcb.vias:
        m = out.get(v.net)
        if m is None:
            continue
        m.vias += 1
    models: dict[tuple[str, float], tuple[LineModel | None, str | None]] = {}
    for (net, layer, width), length in sorted(per.items(), key=lambda kv: (names.index(kv[0][0]), kv[0][1], kv[0][2])):
        if (layer, width) not in models:
            models[(layer, width)] = line_model(stackup, layer, width)
        model, why = models[(layer, width)]
        seg = Segment(layer=layer, width_mm=width, length_mm=length, model=model, reason=why)
        if (net, layer, width) in under and model is not None:
            seg.keepout_mm, seg.keepout_reason, seg.keepout_id = under[(net, layer, width)]
            bound = _bound_t_pd(stackup, layer)
            if bound is not None:
                seg.keepout_t_pd, seg.keepout_bound = bound
        out[net].segments.append(seg)
    for m in out.values():
        if vm is not None:
            m.via_length_mm, t_pd_via = vm
            m.via_delay_s = m.via_length_mm / 1000.0 * t_pd_via
        if stackup is None and m.routed:
            m.problems.append(NO_STACKUP)
        for s in m.segments:
            if s.model is None:
                m.problems.append(f"{s.layer} {s.width_mm:g} mm: {s.reason}")
    _paths(ir, out, library, models, vm)
    return out


def _paths(ir: CircuitIR, out: dict[str, NetMeasure], library: KicadLibrary | None,
           models: dict[tuple[str, float], tuple[LineModel | None, str | None]], vm: tuple[float, float] | None) -> None:
    """Each routed net's longest pad-to-pad path (or why not) - see :mod:`ai_eda.tools.si.paths`."""
    pcb = ir.pcb
    assert pcb is not None
    routed = [m for m in out.values() if m.routed and m.pads >= 2]
    if not routed:
        return
    if library is None:
        for m in routed:
            m.path_problem = "no KiCad library: the pad positions are not known"
        return
    pads = net_pads(ir, library)
    tracks: dict[str, list] = {}
    vias: dict[str, list] = {}
    for t in pcb.tracks:
        tracks.setdefault(t.net, []).append(t)
    for v in pcb.vias:
        vias.setdefault(v.net, []).append(v)
    t_pd: dict[tuple[str, float], tuple[float | None, bool]] = {}
    for (layer, width), (model, _why) in models.items():
        t_pd[(layer, width)] = (None, False) if model is None else (model.t_pd_s_per_m, model.bound)
    for m in routed:
        problems = pads.problems.get(m.net)
        if problems:
            m.path_problem = "; ".join(problems)
            continue
        local = t_pd
        under = [s for s in m.segments if s.keepout_mm > 0.0 and s.keepout_t_pd is not None]
        if under:  # copper under a keep-out: the whole (layer, width) of this net takes the no-plane bound (conservative, module docstring)
            local = dict(t_pd)
            for s in under:
                local[(s.layer, s.width_mm)] = (s.keepout_t_pd, True)
        m.path, m.path_problem = longest_path(m.net, tracks.get(m.net, []), vias.get(m.net, []), pads.pads.get(m.net, {}), pcb.stackup, local,
                                              None if vm is None else vm[1])


def _bound_t_pd(stackup: Stackup | None, layer: str) -> tuple[float, str] | None:
    """``(t_pd, which er)``: the no-plane upper bound of the delay per length on an outer layer, sqrt(er_max)/c0 over the stack's dielectrics.

    With a keep-out clearing the plane the field of the track reaches past
    the prepreg toward the next plane, so its own dielectric's er is no bound
    when the core's is larger (module docstring); ``None`` without a stackup
    or on an inner layer.
    """
    if stackup is None or layer not in stackup.copper_names():
        return None
    d = _outer_dielectric(stackup, layer)
    if d is None:
        return None
    er, k = _bound_er(stackup, d)
    return propagation_delay(er), f"sqrt(er_max)/c0 with pcb.stackup.dielectrics[{k}].er = {er:g} (the largest er of the stack)"


#: the remedy for copper on a layer the stack gives no plane (never for a plane a keep-out cleared)
NO_PLANE_ADVICE = f"{NO_REFERENCE_PLANE} - use pcb_layers=4 or add a plane"


def no_plane_reason(m: NetMeasure) -> str:
    """Why a net whose delay is the no-plane bound has no impedance: the keep-out(s) that cleared its plane, and/or its plane-less layers.

    A keep-out that clears the plane on a board that has one is named with
    its own reason (``impedance is undefined without a reference plane under
    keep-out ANT: ...``) - "use pcb_layers=4 or add a plane" would be the
    wrong advice there; that remedy is given only for copper on a layer the
    stack gives no plane (both when both apply).
    """
    parts = list(m.keepout_reasons)
    if m.plane_less or not parts:
        parts.append(NO_PLANE_ADVICE)
    return "; ".join(parts)


def undefined_impedance_reason(model: LineModel | None, why: str | None) -> str:
    """The sentence an impedance check gives for a segment without Z0 on a layer the stack gives no plane (a keep-out's is its own reason)."""
    if model is None:
        return why or NO_STACKUP
    reason = model.reason or NO_REFERENCE_PLANE
    if reason.startswith(NO_REFERENCE_PLANE):
        detail = reason[len(NO_REFERENCE_PLANE):].lstrip(": ")
        return NO_PLANE_ADVICE + (f" ({detail})" if detail else "")
    return reason


__all__ = [
    "KEEPOUT_NO_PLANE",
    "NOT_DRC",
    "NO_PLANE_ADVICE",
    "SI_TOOL",
    "SI_VERSION",
    "LineModel",
    "NetLine",
    "NetMeasure",
    "Segment",
    "line_model",
    "measure_nets",
    "no_plane_reason",
    "plane_keepout",
    "undefined_impedance_reason",
    "via_model",
]
