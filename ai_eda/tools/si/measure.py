"""What the routed copper of each net is, electrically: length, via barrels, propagation delay, impedance per segment.

Invariant: every number here is computed from the IR copper (``ir.pcb.tracks``
/ ``ir.pcb.vias``), the board's stackup (``ir.pcb.stackup``) and the registered
transmission-line calculators (:mod:`ai_eda.tools.calc.tline`) - nothing is
estimated from a picture and nothing is guessed. What cannot be computed is
said, never replaced by a number:

* no stackup -> no delay and no impedance (:data:`~ai_eda.tools.calc.tline.NO_STACKUP`);
* a track on a layer whose adjacent copper is not a plane has no impedance
  (:data:`~ai_eda.tools.calc.tline.NO_REFERENCE_PLANE`). Its delay is
  bounded, not computed: a line's field lies partly in the dielectric and
  partly in air, so its effective permittivity is at most the dielectric's
  er and its delay per length at most ``calc.tline.tpd(er)`` = sqrt(er)/c0.
  The bound is conservative for the critical-length rule (it makes a line
  look longer, never shorter) and every result that uses it says so
  (``bound``);
* over a plane the segment is a microstrip: Z0 and e_eff from
  ``calc.tline.microstrip.*`` (Hammerstad & Jensen) at the routed width, the
  delay per length ``calc.tline.tpd(e_eff)``.

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
    er = float(stackup.dielectrics[d].er.value)
    return LineModel(
        layer=layer, width_mm=width_mm, z0_ohm=None, e_eff=er, t_pd_s_per_m=propagation_delay(er), bound=True, reason=why,
        ids=(f"pcb.stackup.dielectrics[{d}].er",),
    ), None


def via_model(stackup: Stackup | None) -> tuple[float, float] | None:
    """``(barrel length mm, delay per length s/m)`` of a through via, or ``None`` without a stackup."""
    if stackup is None:
        return None
    er_max = max(float(d.er.value) for d in stackup.dielectrics)
    return stackup.span_mm("F.Cu", "B.Cu"), propagation_delay(er_max)


@dataclass
class Segment:
    """All of a net's copper on one layer at one width: total length and its line model."""

    layer: str
    width_mm: float
    length_mm: float
    model: LineModel | None
    reason: str | None = None

    @property
    def delay_s(self) -> float | None:
        return None if self.model is None else self.length_mm / 1000.0 * self.model.t_pd_s_per_m


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
        """Whether any part of the delay is the no-plane upper bound."""
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
                 "reason": s.reason or (None if s.model is None else s.model.reason)}
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
    for t in pcb.tracks:
        m = out.get(t.net)
        if m is None:
            continue
        length = math.hypot(t.end[0] - t.start[0], t.end[1] - t.start[1])
        m.track_length_mm += length
        key = (t.net, t.layer, float(t.width_mm))
        per[key] = per.get(key, 0.0) + length
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
        out[net].segments.append(Segment(layer=layer, width_mm=width, length_mm=length, model=model, reason=why))
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
        m.path, m.path_problem = longest_path(m.net, tracks.get(m.net, []), vias.get(m.net, []), pads.pads.get(m.net, {}), pcb.stackup, t_pd,
                                              None if vm is None else vm[1])


def undefined_impedance_reason(model: LineModel | None, why: str | None) -> str:
    """The sentence an impedance check gives for a segment without Z0."""
    if model is None:
        return why or NO_STACKUP
    reason = model.reason or NO_REFERENCE_PLANE
    if reason.startswith(NO_REFERENCE_PLANE):
        detail = reason[len(NO_REFERENCE_PLANE):].lstrip(": ")
        return f"{NO_REFERENCE_PLANE} - use pcb_layers=4 or add a plane" + (f" ({detail})" if detail else "")
    return reason


__all__ = [
    "NOT_DRC",
    "SI_TOOL",
    "SI_VERSION",
    "LineModel",
    "NetLine",
    "NetMeasure",
    "Segment",
    "line_model",
    "measure_nets",
    "undefined_impedance_reason",
    "via_model",
]
