"""From the IR's net classes (``ir.si``) to the router's per-net rules (:class:`~ai_eda.tools.routing.maze.NetRule`).

Invariant: every number a rule carries is a registered calculator's output
over the IR's own traced values, a class constraint as the IR states it, or
the router's own board value - never a guess. What a class cannot be given
(no stackup, no reference plane under a routing layer, a delay budget with
no delay per length to convert it) is left out of the rule and named in a
note, so the router routes that net at the board rules and the ``si.*``
checks say why it is not controlled.

Per class (the rule applies to every net the class routes: its declared nets
and the nets promoted into it):

* **controlled impedance** (``target_z0_ohm``): the track width is
  ``calc.tline.width_for_z0.microstrip`` over the plane under each routing
  layer (``F.Cu`` / ``B.Cu``, :func:`~ai_eda.tools.calc.tline.line_geometry`),
  rounded **up** to :data:`WIDTH_STEP_MM` and never below the fab minimum;
  where a pad pitch cannot take that width the track may narrow to the
  board's own width (``neckdown_width_mm``) near those pads. The two outer
  layers of a symmetric stack give one width; when they differ the
  ``F.Cu`` width is used and the note says so (``si.impedance`` judges every
  segment on its own layer). Without a stackup or without a plane under a
  routing layer the class gets no width: impedance is undefined there.
* **minimum width** (``min_width_mm``, e.g. IPC-2221 current capacity): the
  width is ``max(board width, min_width rounded up)`` - a rule only when it
  is wider than the board's.
* **length / delay budget**: ``max_length_mm`` as stated; ``max_delay_s``
  converted to a length with the class's *largest* delay per length over
  the routing layers (at the class width over a plane, else the no-plane
  upper bound sqrt(er)/c0, :mod:`ai_eda.tools.si.measure`), so the length
  budget is never looser than the delay budget; the tighter of the two
  applies. A via counts its barrel (the stackup's F.Cu..B.Cu span).
* **match group** (``match_group`` + ``max_skew_s``): the skew converted to
  length the same way.
* **declared differential pairs** (``pairs`` with ``target_zdiff_ohm``): the
  pair width is the class's controlled width (or the board width when the
  class has no ``target_z0_ohm``) and the gap is
  ``calc.tline.edge_coupled_microstrip.s_for_zdiff`` for it, rounded to
  :data:`WIDTH_STEP_MM`; a gap below the board clearance re-solves the width
  for the gap at the clearance (``w_for_zdiff``). Without a plane a pair is
  routed as two nets and named in a note.

A promoted net takes the controlled class's rule; the length / delay budget
of the class that declares it still applies (the tighter one wins).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ai_eda.ir import CircuitIR, NetClass, SIConstraints
from ai_eda.tools.calc.tline import (
    NO_REFERENCE_PLANE,
    NO_STACKUP,
    TLineRangeError,
    edge_coupled_microstrip,
    line_geometry,
    solve_coupled_spacing,
    solve_coupled_width,
    solve_microstrip_width,
)
from ai_eda.tools.routing.maze import LAYERS, NetRule, RoutingParams, effective_params
from ai_eda.tools.si.measure import line_model

#: the resolution a derived track width / pair gap is rounded to (mm): widths up, gaps to the nearest step
WIDTH_STEP_MM = 0.01


def round_up(value: float, step: float = WIDTH_STEP_MM) -> float:
    """``value`` rounded up to a multiple of ``step`` (a float spelled with at most the step's decimals)."""
    k = math.ceil(value / step - 1e-9)
    return round(k * step, 6)


def round_near(value: float, step: float = WIDTH_STEP_MM) -> float:
    return round(round(value / step) * step, 6)


@dataclass
class ClassRule:
    """What one class maps to (before it is copied onto each of its nets), with how each number was obtained."""

    name: str
    width_mm: float | None = None
    neckdown_width_mm: float | None = None
    max_length_mm: float | None = None
    via_length_mm: float | None = None
    match_group: str | None = None
    max_skew_mm: float | None = None
    pair_width_mm: float | None = None
    pair_spacing_mm: float | None = None
    pair_uncoupled_max_mm: float | None = None
    pair_max_skew_mm: float | None = None
    #: human sentences: how the numbers were derived, and what could not be
    derivation: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    #: the exact calculator outputs before rounding (for the checks' details)
    exact: dict[str, float] = field(default_factory=dict)


@dataclass
class SIRules:
    """The router's rules for a board, per net, and the per-class derivation (:class:`ClassRule`)."""

    rules: dict[str, NetRule] = field(default_factory=dict)
    classes: dict[str, ClassRule] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def signature(self) -> tuple:
        """A value that is equal for two rule sets the router would route identically."""
        return tuple(sorted((k, v) for k, v in self.rules.items()))


def _t_pd_max(ir: CircuitIR, width: float) -> tuple[float | None, str]:
    """The largest delay per length (s/m) of a ``width`` track over the routing layers, and how it was obtained."""
    stackup = ir.pcb.stackup if ir.pcb is not None else None
    worst: float | None = None
    how: list[str] = []
    for layer in LAYERS:
        model, why = line_model(stackup, layer, width)
        if model is None:
            return None, why or NO_STACKUP
        if worst is None or model.t_pd_s_per_m > worst:
            worst = model.t_pd_s_per_m
        how.append(f"{layer} {'sqrt(er)/c0 upper bound' if model.bound else 'microstrip sqrt(e_eff)/c0'} {model.t_pd_s_per_m * 1e9:.4f} ps/mm")
    return worst, "; ".join(how)


def controlled_width(ir: CircuitIR, target_z0: float) -> tuple[float | None, dict[str, float], str]:
    """``(exact width, per-layer exact widths, how)`` for ``target_z0`` over the plane under each routing layer, or ``(None, {}, why)``."""
    stackup = ir.pcb.stackup if ir.pcb is not None else None
    if stackup is None:
        return None, {}, NO_STACKUP
    widths: dict[str, float] = {}
    how: list[str] = []
    for layer in LAYERS:
        geom, why = line_geometry(stackup, layer)
        if geom is None:
            reason = why or NO_REFERENCE_PLANE
            return None, {}, f"{layer}: {reason}"
        try:
            w = solve_microstrip_width(target_z0, float(geom.h.value), float(geom.t.value), float(geom.er.value))
        except (TLineRangeError, ValueError) as e:
            return None, {}, f"{layer}: calc.tline.width_for_z0.microstrip refuses {target_z0:g} ohm: {e}"
        widths[layer] = w
        how.append(
            f"{layer}: calc.tline.width_for_z0.microstrip({target_z0:g} ohm, h {float(geom.h.value):g} mm [{geom.h_id}], "
            f"t {float(geom.t.value):g} um [{geom.t_id}], er {float(geom.er.value):g} [{geom.er_id}]) = {w:.6f} mm over {geom.reference_layer} ({geom.reference_net})"
        )
    return widths[LAYERS[0]], widths, "; ".join(how)


def class_rule(ir: CircuitIR, cls: NetClass, p: RoutingParams) -> ClassRule:
    """The rule of one class (module docstring); ``p`` are the effective board parameters."""
    out = ClassRule(name=cls.name)
    mfg = ir.pcb.manufacturing if ir.pcb is not None else None
    fab_min = float(mfg.min_track_width_mm.value) if mfg is not None and mfg.min_track_width_mm is not None else None
    if cls.target_z0_ohm is not None:
        target = float(cls.target_z0_ohm.value)
        exact, per_layer, how = controlled_width(ir, target)
        if exact is None:
            out.missing.append(f"no controlled width for {target:g} ohm: {how}")
        else:
            widths = {layer: round_up(w) for layer, w in per_layer.items()}
            w = widths[LAYERS[0]]
            if len(set(widths.values())) > 1:
                out.derivation.append(f"the outer layers need different widths {widths}; {LAYERS[0]}'s {w:g} mm is used (si.impedance judges each layer)")
            if fab_min is not None and w < fab_min:
                out.derivation.append(f"{w:g} mm raised to the fab minimum {fab_min:g} mm")
                w = fab_min
            out.width_mm = w
            out.exact["width_for_z0_mm"] = exact
            out.derivation.append(f"width {w:g} mm = {exact:.6f} mm rounded up to {WIDTH_STEP_MM:g} mm ({how})")
            if w > p.track_width_mm:
                out.neckdown_width_mm = p.track_width_mm
                out.derivation.append(f"neck-down to the board width {p.track_width_mm:g} mm where a pad pitch cannot take {w:g} mm")
    if cls.min_width_mm is not None:
        need = round_up(float(cls.min_width_mm.value))
        base = out.width_mm if out.width_mm is not None else p.track_width_mm
        if need > base:
            out.width_mm = need
            out.derivation.append(f"minimum width {float(cls.min_width_mm.value):.6f} mm ({cls.min_width_mm.provenance.tool or cls.min_width_mm.provenance.kind.value}) rounded up to {need:g} mm")
            if out.neckdown_width_mm is None and need > p.track_width_mm:
                out.neckdown_width_mm = p.track_width_mm
        else:
            out.derivation.append(
                f"minimum width {float(cls.min_width_mm.value):.6f} mm ({cls.min_width_mm.provenance.tool or cls.min_width_mm.provenance.kind.value}) "
                f"is not above the {base:g} mm the class is routed at: no wider rule"
            )
    t_pd: float | None = None
    t_pd_how = ""
    if cls.max_delay_s is not None or cls.max_skew_s is not None:
        t_pd, t_pd_how = _t_pd_max(ir, out.width_mm or p.track_width_mm)
    lengths: list[float] = []
    if cls.max_length_mm is not None:
        lengths.append(float(cls.max_length_mm.value))
        out.derivation.append(f"length budget {float(cls.max_length_mm.value):g} mm (max_length_mm)")
    if cls.max_delay_s is not None:
        if t_pd is None:
            out.missing.append(f"max_delay_s not converted to a length: {t_pd_how}")
        else:
            converted = float(cls.max_delay_s.value) / t_pd * 1000.0
            lengths.append(converted)
            out.derivation.append(f"delay budget {float(cls.max_delay_s.value) * 1e12:g} ps = {converted:.4f} mm at the largest t_pd ({t_pd_how})")
    if lengths:
        out.max_length_mm = min(lengths)
    if cls.match_group is not None and cls.max_skew_s is not None:
        if t_pd is None:
            out.missing.append(f"match group {cls.match_group}: max_skew_s not converted to a length: {t_pd_how}")
        else:
            out.match_group = cls.match_group
            out.max_skew_mm = float(cls.max_skew_s.value) / t_pd * 1000.0
            out.derivation.append(f"match group {cls.match_group}: skew {float(cls.max_skew_s.value) * 1e12:g} ps = {out.max_skew_mm:.4f} mm")
    if cls.pairs:
        _pair_rule(ir, cls, p, out)
    stackup = ir.pcb.stackup if ir.pcb is not None else None
    if stackup is not None and (out.max_length_mm is not None or out.match_group is not None):
        out.via_length_mm = round(stackup.span_mm("F.Cu", "B.Cu"), 6)
    return out


def _pair_rule(ir: CircuitIR, cls: NetClass, p: RoutingParams, out: ClassRule) -> None:
    zdiff = float(cls.target_zdiff_ohm.value)  # type: ignore[union-attr]  # the model requires it with pairs
    stackup = ir.pcb.stackup if ir.pcb is not None else None
    geom, why = (None, NO_STACKUP) if stackup is None else line_geometry(stackup, LAYERS[0])
    if geom is None:
        out.missing.append(f"pairs {', '.join(pr.name for pr in cls.pairs)} routed as two nets: {why}")
        return
    h, t, er = float(geom.h.value), float(geom.t.value), float(geom.er.value)
    w = out.width_mm if out.width_mm is not None and cls.target_z0_ohm is not None else p.track_width_mm
    try:
        s = round_near(solve_coupled_spacing(zdiff, w, h, t, er))
        how = f"s = calc.tline.edge_coupled_microstrip.s_for_zdiff({zdiff:g} ohm, w {w:g} mm) rounded to {WIDTH_STEP_MM:g} mm"
    except (TLineRangeError, ValueError) as e:
        s, how = None, f"no gap for w {w:g} mm: {e}"
    if s is None or s < p.clearance_mm:
        s = round_up(p.clearance_mm)
        try:
            w = round_up(solve_coupled_width(zdiff, s, h, t, er))
            how += f"; the gap is held at the board clearance {s:g} mm and w = calc.tline.edge_coupled_microstrip.w_for_zdiff({zdiff:g} ohm, s {s:g} mm) rounded up"
        except (TLineRangeError, ValueError) as e:
            out.missing.append(f"pairs {', '.join(pr.name for pr in cls.pairs)} routed as two nets: no w / s for {zdiff:g} ohm: {e}")
            return
    try:
        z = edge_coupled_microstrip(w, s, h, t, er).z_diff
    except (TLineRangeError, ValueError) as e:
        out.missing.append(f"pairs {', '.join(pr.name for pr in cls.pairs)} routed as two nets: {e}")
        return
    out.pair_width_mm, out.pair_spacing_mm = w, s
    out.exact["pair_z_diff_ohm"] = z
    out.derivation.append(f"pair w {w:g} mm, s {s:g} mm -> Z_diff {z:.3f} ohm ({how})")
    t_pd, t_pd_how = _t_pd_max(ir, w)
    if cls.pair_uncoupled_max_mm is not None:
        out.pair_uncoupled_max_mm = float(cls.pair_uncoupled_max_mm.value)
    if cls.pair_max_skew_s is not None:
        if t_pd is None:
            out.missing.append(f"pair_max_skew_s not converted to a length: {t_pd_how}")
        else:
            out.pair_max_skew_mm = float(cls.pair_max_skew_s.value) / t_pd * 1000.0


def _net_rule(cr: ClassRule, net: str, partner: str | None, extra_length: float | None, via_length: float | None) -> NetRule:
    if partner is not None:
        width = cr.pair_width_mm
    else:
        width = cr.width_mm
    lengths = [x for x in (cr.max_length_mm, extra_length) if x is not None]
    max_length = min(lengths) if lengths else None
    return NetRule(
        net_class=cr.name,
        width_mm=width,
        neckdown_width_mm=cr.neckdown_width_mm if partner is None and width is not None and cr.neckdown_width_mm is not None and cr.neckdown_width_mm < width else None,
        max_length_mm=max_length,
        via_length_mm=(cr.via_length_mm if cr.via_length_mm is not None else via_length) if max_length is not None or cr.match_group is not None else None,
        match_group=cr.match_group if partner is None else None,
        max_skew_mm=cr.max_skew_mm if partner is None else None,
        pair_partner=partner,
        pair_spacing_mm=cr.pair_spacing_mm if partner is not None else None,
        pair_uncoupled_max_mm=cr.pair_uncoupled_max_mm if partner is not None else None,
        pair_max_skew_mm=cr.pair_max_skew_mm if partner is not None else None,
    )


def net_rules(ir: CircuitIR, params: RoutingParams | None = None) -> SIRules:
    """The router's rules for ``ir`` (module docstring); an IR without ``ir.si`` (or whose classes constrain nothing) gets none."""
    out = SIRules()
    si: SIConstraints | None = ir.si
    if si is None or ir.pcb is None:
        return out
    p, _ = effective_params(ir, params)
    names = {n.name for n in ir.nets}
    routed_by = {n: c.name for n in names if (c := si.class_of(n)) is not None}
    for cls in si.net_classes:
        cr = class_rule(ir, cls, p)
        out.classes[cls.name] = cr
        if cls.name in routed_by.values():  # what a class without nets cannot get is no news
            out.notes.extend(f"net class {cls.name}: {m}" for m in cr.missing)
    stackup = ir.pcb.stackup
    via_length = round(stackup.span_mm("F.Cu", "B.Cu"), 6) if stackup is not None else None
    paired: dict[str, str] = {}
    for cls in si.net_classes:
        cr = out.classes[cls.name]
        if cr.pair_width_mm is None:
            continue
        for pr in cls.pairs:
            paired[pr.p], paired[pr.n] = pr.n, pr.p
    for net in [n.name for n in ir.nets]:
        cls = si.class_of(net)
        if cls is None:
            continue
        cr = out.classes[cls.name]
        extra: float | None = None
        declared = si.declared_class_of(net)
        if declared is not None and declared.name != cls.name:  # a promoted net keeps the length budget of the class that declares it
            extra = out.classes[declared.name].max_length_mm
        partner = paired.get(net) if paired.get(net) in names else None
        rule = _net_rule(cr, net, partner, extra, via_length)
        if not rule.is_empty():
            out.rules[net] = rule
    return out


def class_widths(ir: CircuitIR, params: RoutingParams | None = None) -> dict[str, ClassRule]:
    """The per-class derivation alone (what the checks compare the copper with)."""
    return net_rules(ir, params).classes


__all__ = ["WIDTH_STEP_MM", "ClassRule", "SIRules", "class_rule", "class_widths", "controlled_width", "net_rules", "round_near", "round_up"]
