"""Signal-integrity checks of the routed copper against ``ir.si`` (IR geometry + registered calculators, never DRC).

Invariant: every verdict here is a deterministic comparison of numbers the
IR copper, the stackup and the registered transmission-line calculators give
(:mod:`ai_eda.tools.si.measure`) with the constraints the IR states
(:class:`~ai_eda.ir.SIConstraints`). A number that cannot be computed makes
the row NOT_VERIFIED naming why - no stackup; "impedance is undefined without
a reference plane" on a layer whose neighbour is not a plane; a datasheet
timing term nobody grounded - and never becomes a PASS. A bound is used only
in the direction it is valid: without a plane the delay per length is the
upper bound sqrt(er)/c0, so a budget it meets is met, one it breaks is
NOT_VERIFIED (the real delay is smaller). The fab's measured impedance is the
only real one (``details["kind"]``, every message ends with the note).

A track with a ``mid`` point is a circular arc (:mod:`ai_eda.ir.arc`) and is
judged as one, never as the straight line between its ends: its arc length
in every length, delay and impedance row (:mod:`ai_eda.tools.si.measure`),
a neck-down test sampled on the arc itself, never a segment of an escape
stub, never part of a straight run, of a coupled piece or of a parallel run -
the coupled-line formulas are about straight parallel conductors, so curved
copper beside another net is named, not computed.

The validator applies only to a design that states SI constraints
(``ir.si`` is set): an IR without them gets no result, so older designs keep
their verdicts. Results (tool :data:`~ai_eda.tools.si.measure.SI_TOOL`; version
:func:`~ai_eda.tools.si.measure.si_version`: 0.3 on a board with an arc
track, 0.2 on one with a blind / buried / micro via, 0.1 on every other):

* ``si.critical_length`` - every routed net's line delay against its
  class's driver edge (:func:`ai_eda.tools.si.promote.critical_rows`: the
  line is the longest pad-to-pad path, or a 2-pad net's copper): PASS rows
  "electrically short - transmission-line effects negligible by the stated
  rule (not a simulation)"; a long net over a plane is a NOT_APPLICABLE row
  of this rule "electrically long ... judged by spice.si.<net>" (the rule
  found the opposite of its claim there: the verdict is the SPICE check's,
  never a PASS row here); a long net without a plane (or with no delay) is
  NOT_VERIFIED "needs impedance control: impedance is undefined without a
  reference plane - use pcb_layers=4 or add a plane"; a net whose whole
  copper exceeds l_crit but whose pad-to-pad path was not extracted is
  NOT_VERIFIED "possibly long" (the total is only an upper bound of every
  path); supply / ground nets and nets whose class states no edge are
  NOT_APPLICABLE rows. The result is the worst row; when every judged net is
  long over a plane it is NOT_APPLICABLE naming them (no net is short).
* ``si.rf_length`` - only when the design has RF nets (a net of kind ``rf``
  or a class stating ``rf_frequency_hz``, :func:`ai_eda.tools.si.rf.rf_nets`):
  each RF net's line (the same line, t_pd and bound flag as the rule above)
  against l_crit = ``rf_length_fraction`` x lambda_g at its frequency (the
  class's ``rf_frequency_hz``, else the confirmed ``carrier_frequency``).
  PASS rows "electrically short at <f> (<fraction> lambda_g rule, not a
  simulation)" (valid with the no-plane bound too: l_crit is then a lower
  bound); a line longer over a plane is a NOT_APPLICABLE row "judged by
  si.impedance.<class> / domain.rf.impedance" when its class states a
  target Z0, NOT_VERIFIED "needs impedance control" when it does not;
  longer only by a bound (no plane, or a multi-pad net's whole copper) is
  NOT_VERIFIED "possibly long"; no frequency, no stated fraction or no
  routed copper is NOT_VERIFIED naming it. Nothing is promoted or re-routed.
* ``si.impedance.<class>`` - per routed segment of the class's nets, Z0 of
  the routed width on its layer over its plane (``calc.tline.microstrip.z0``)
  within ``target_z0_ohm`` +/- ``z0_tol_rel``. A track narrower than the
  class's controlled width is a neck-down - named with its length and not
  judged - only when the router necked it down: its provenance records the
  net's rule with that ``neckdown_width`` and (with the KiCad library, which
  gives the pad boxes) it lies within the rule's ``neckdown_radius`` plus one
  grid step of one of the net's pads - or it is a segment of an escape stub
  the router recorded (routing.maze 0.6: a fine-pitch pad joined to the grid
  by a stub no wider than the pad, whose provenance names the pad, the width
  and the stub's points, and the track is one of its segments). Every other
  narrower track is judged at its own width (a promoted net the router left
  at the board width FAILs). A track whose copper lies in a keep-out that forbids the zone of
  its plane layer (:func:`ai_eda.tools.si.measure.plane_keepout`) has no
  reference plane there: its row is NOT_VERIFIED with that keep-out named,
  never judged as a microstrip. NOT_APPLICABLE for a class without a target
  or without nets.
* ``si.width.<class>`` - every track of the class's nets at least
  ``min_width_mm`` (a class with one). A member net with two or more pads
  and no track is a NOT_VERIFIED row (the width it needs is not judged; a
  net joined only through a plane zone says so) - never NOT_APPLICABLE,
  which is kept for a class none of whose members needs copper.
* ``si.length.<class>`` / ``si.delay.<class>`` - each net the class declares
  (the default class: every net no class lists): routed length (tracks + via
  barrels from the stackup) against ``max_length_mm``, delay against
  ``max_delay_s``.
* ``si.skew.<group>`` - the delay spread of a match group against its
  ``max_skew_s`` (NOT_VERIFIED when a member's delay is only bounded).
* ``si.diff.<P/N>`` - a declared pair: Z_diff of the coupled section
  (parallel P / N segments on one layer; ``calc.tline.edge_coupled_microstrip.z_diff``
  at the measured width and gap) within tolerance, each net's uncoupled
  length - its tracks outside the coupled overlap: breakouts, mitred
  corners, compensation bumps (:mod:`ai_eda.tools.routing.coupling`, the
  definition the router enforces) - against ``pair_uncoupled_max_mm``, the
  P / N delay difference against ``pair_max_skew_s``. An arc track is never
  a coupled piece: its whole arc length is uncoupled copper of its net (the
  row says how much of it is curved), and a pair with an arc and no straight
  coupled section is NOT_VERIFIED - what its curved copper does is not
  judged.
* ``si.crosstalk`` - its own tool and version (:data:`XT_TOOL` :data:`XT_VERSION`,
  :data:`XT_ARC_VERSION` on a board with an arc track - :func:`xt_version` -,
  never re-stamped): one row per IR net with the other nets' copper beside
  it within s <= 10 h and the matched-termination closed forms of each
  parallel run (:func:`crosstalk_result`, :mod:`ai_eda.tools.calc.xtalk`).
  They rank and explain: the check never says PASS or FAIL, because every
  opened form assumes Z0 at both victim ends and an ai_eda net is driven
  through R_drive into C_load; what no formula covers is named per row. A
  victim whose every neighbour is computed and whose coupled-line decks can
  be built is a NOT_APPLICABLE row "judged by spice.xtalk.<victim>"
  (:mod:`ai_eda.tools.spice.xtalk_check`), otherwise NOT_VERIFIED naming why.
* ``si.timing.<path>`` - setup and hold margins of a timing path with the
  routed flight times: T = 1 / f_clk, t_lc = capture_fraction * T,
  setup = t_lc + t_flight(clock) - t_co_max - t_flight(data) - t_su_min,
  hold = (T - t_lc) + t_co_min + t_flight(data) - t_flight(clock) - t_h_min.
  A flight time is known only as the interval [0, the net's whole routed
  delay] (the path from the launching to the capturing pin is not
  identified), so the margins are intervals: PASS when the worst case is
  >= 0, FAIL when even the best case is < 0, NOT_VERIFIED in between;
  NOT_VERIFIED naming every missing term (``U1.t_su`` ...) otherwise. A term
  named on a connector (``J2.t_co``: the ISP header) says that the part
  launching the edge is off-board, behind that connector.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ai_eda.ir import CircuitIR, NetClass, NetKind, SIConstraints, TimingPath, Track, Traced, ValidationResult, ValidationStatus, track_length, worst_status
from ai_eda.ir.arc import arc_point_at
from ai_eda.ir.si import FACT_REF_RE, TIMING_TERMS
from ai_eda.tools.calc.tline import COUPLED_G_RANGE, NO_STACKUP, TLineRangeError, edge_coupled_microstrip, line_geometry, propagation_delay
from ai_eda.tools.calc.xtalk import fext_rel, kb_eq8, l_sat_mm, next_rel
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.routing.coupling import COUPLED_GAP_FACTOR, NeighbourIndex, coupled_pieces, uncoupled_lengths
from ai_eda.tools.routing.maze import ROUTER_ESCAPE_VERSION, escape_entry
from ai_eda.tools.si.measure import (
    NOT_DRC,
    NO_PLANE_ADVICE,
    SI_TOOL,
    SI_VERSION,
    NetMeasure,
    line_model,
    measure_nets,
    no_plane_reason,
    plane_keepout,
    si_version,
    undefined_impedance_reason,
)
from ai_eda.tools.si.driver import driver_value
from ai_eda.tools.si.paths import NetPads, PadBox, net_pads
from ai_eda.tools.si.promote import critical_rows
from ai_eda.tools.si.rf import RF_IMPEDANCE_CHECK, RF_LENGTH_CHECK, format_hz, rf_length_rows, rf_nets
from ai_eda.tools.si.rules import controlled_width, round_up
from ai_eda.validation.base import ValidationContext, Validator
from ai_eda.validation.registry import default_registry

if TYPE_CHECKING:  # imported where used: the SPICE deck module imports this one
    from ai_eda.tools.spice.xtalk_check import VictimPlan, XtPiece

S = ValidationStatus
KIND = "ir_geometry+calculators"
#: a part of a symbol library with this prefix is a connector (its timing terms come from the off-board part behind it)
CONNECTOR_LIBRARY_PREFIX = "Connector"
CRITICAL_CHECK = "si.critical_length"
SHORT_TEXT = "electrically short - transmission-line effects negligible by the stated rule (not a simulation)"
_TOL = 1e-9


def _result(check_id: str, status: S, message: str, **details: Any) -> ValidationResult:
    return ValidationResult(check_id=check_id, status=status, message=f"{message} ({NOT_DRC})", tool=SI_TOOL, tool_version=SI_VERSION, details={"kind": KIND, **details})


def _worst(rows: list[dict]) -> S:
    return worst_status(S(r["status"]) for r in rows) if rows else S.NOT_APPLICABLE


def _declared_nets(ir: CircuitIR, si: SIConstraints, cls: NetClass) -> list[str]:
    """The nets whose budgets ``cls`` states: its own list, or - for the default class - every IR net no class lists."""
    return [n.name for n in ir.nets if (c := si.declared_class_of(n.name)) is not None and c.name == cls.name]


def _members(ir: CircuitIR, si: SIConstraints, cls: NetClass) -> list[str]:
    """The nets ``cls`` routes (declared, promoted, and for the default class every unlisted net), in IR order."""
    return [n.name for n in ir.nets if (c := si.class_of(n.name)) is not None and c.name == cls.name]


def _pads(ir: CircuitIR, net: str) -> int:
    n = ir.net(net)
    return 0 if n is None else len({(p.component_ref, p.pin_number) for p in n.pins})


# --------------------------------------------------------------------------- critical length


def critical_result(ir: CircuitIR, measures: dict[str, NetMeasure]) -> ValidationResult:
    rows: list[dict] = []
    long_spice: list[str] = []
    uncontrolled: list[str] = []
    cleared: list[str] = []  # long nets whose reference plane a keep-out clears (named, never "use pcb_layers=4")
    possibly: list[str] = []
    for r in critical_rows(ir, measures):
        row = r.as_dict()
        where = f" ({r.measure})" if r.measure else ""
        if r.status == "not_applicable":
            row["status"] = S.NOT_APPLICABLE.value
        elif r.status == "unknown":
            if r.reason == "no routed copper" and _pads(ir, r.net) < 2:
                row["status"], row["reason"] = S.NOT_APPLICABLE.value, "fewer than two pads: nothing routed"
            else:
                row["status"] = S.NOT_VERIFIED.value
        elif r.status == "possibly_long":
            row["status"] = S.NOT_VERIFIED.value
            possibly.append(r.net)
        elif r.status == "short":
            row["status"], row["reason"] = S.PASS.value, SHORT_TEXT + (" (with the no-plane upper bound of t_pd)" if r.bound else "")
        elif r.bound:
            m = measures.get(r.net)
            row["status"] = S.NOT_VERIFIED.value
            row["reason"] = (f"electrically long ({r.length_mm:.3f} mm > l_crit {r.l_crit_mm:.3f} mm by the upper bound of t_pd{where}): needs impedance control, "
                             f"but {NO_PLANE_ADVICE if m is None else no_plane_reason(m)}")
            if m is not None and m.keepout_ids:
                row["keepouts"] = m.keepout_ids
                cleared.append(r.net)
                if m.plane_less:
                    uncontrolled.append(r.net)
            else:
                uncontrolled.append(r.net)
        else:
            row["status"] = S.NOT_APPLICABLE.value
            row["reason"] = f"electrically long ({r.length_mm:.3f} mm > l_crit {r.l_crit_mm:.3f} mm{where}): over a plane, judged by spice.si.{r.net}, not by this rule"
            long_spice.append(r.net)
        rows.append(row)
    status = _worst(rows)
    short = [r["net"] for r in rows if r["status"] == S.PASS.value]
    details = {"nets": rows, "long_over_plane": long_spice, "long_without_plane": uncontrolled, "long_under_keepout": cleared, "possibly_long": possibly,
               "fraction": None if ir.si is None or ir.si.critical_fraction is None else float(ir.si.critical_fraction.value)}
    if status == S.NOT_APPLICABLE:
        if long_spice:
            return _result(CRITICAL_CHECK, status, f"no net is electrically short: {len(long_spice)} electrically long net(s) over a plane, "
                           f"judged by spice.si: {', '.join(long_spice)}", **details)
        return _result(CRITICAL_CHECK, status, "no routed net with a driver edge", **details)
    parts = [f"{len(short)} net(s) {SHORT_TEXT}"]
    if long_spice:
        parts.append(f"{len(long_spice)} electrically long net(s) over a plane, judged by spice.si: {', '.join(long_spice)}")
    if uncontrolled:
        parts.append(f"{len(uncontrolled)} electrically long net(s) need impedance control but impedance is undefined without a reference plane "
                     f"- use pcb_layers=4 or add a plane: {', '.join(uncontrolled)}")
    if cleared:
        parts.append(f"{len(cleared)} electrically long net(s) need impedance control but a keep-out clears their reference plane (impedance is "
                     f"undefined there): " + ", ".join(f"{n} (keep-out {', '.join(measures[n].keepout_ids)})" for n in cleared))
    if possibly:
        parts.append(f"{len(possibly)} net(s) possibly long (the whole copper exceeds l_crit, no pad-to-pad path extracted; not judged): {', '.join(possibly)}")
    other = [f"{r['net']}: {r['reason']}" for r in rows
             if r["status"] == S.NOT_VERIFIED.value and r["net"] not in uncontrolled and r["net"] not in cleared and r["net"] not in possibly]
    if other:
        parts.append(f"{len(other)} net(s) not judged: " + "; ".join(other))
    return _result(CRITICAL_CHECK, status, "; ".join(parts), **details)


# --------------------------------------------------------------------------- RF electrical length


def rf_length_result(ir: CircuitIR, measures: dict[str, NetMeasure]) -> ValidationResult:
    """``si.rf_length``: every RF net's line against fraction x lambda_g at its frequency (module docstring, :mod:`ai_eda.tools.si.rf`)."""
    si = ir.si
    rows: list[dict] = []
    for r in rf_length_rows(ir, measures):
        row = r.as_dict()
        cls = si.net_class(r.net_class) if si is not None and r.net_class is not None else None
        if r.status == "short":
            row["status"] = S.PASS.value
        elif r.status == "not_applicable":
            row["status"] = S.NOT_APPLICABLE.value
        elif r.status == "long" and cls is not None and cls.target_z0_ohm is not None:
            row["status"] = S.NOT_APPLICABLE.value
            row["reason"] = f"electrically long at {format_hz(float(r.f_hz))}: judged by si.impedance.{cls.name} / {RF_IMPEDANCE_CHECK} ({r.reason})"
        elif r.status == "long":
            row["status"] = S.NOT_VERIFIED.value
            owner = f"class {cls.name}" if cls is not None else f"{r.net} (in no net class)"
            row["reason"] = f"needs impedance control: {owner} carries RF at {format_hz(float(r.f_hz))} but states no target_z0_ohm ({r.reason})"
        else:  # possibly_long, unknown
            row["status"] = S.NOT_VERIFIED.value
        rows.append(row)
    status = _worst(rows)
    fraction = None if si is None or si.rf_length_fraction is None else float(si.rf_length_fraction.value)
    details = {"nets": rows, "fraction": fraction, "rule": "l_crit = fraction * lambda_g, lambda_g = 1 / (f t_pd) (calc.tline t_pd; a rule, not a simulation)",
               "promotes": "nothing (the router is unchanged; the rule informs)"}
    short = [r["net"] for r in rows if r["status"] == S.PASS.value]
    parts = [f"{len(short)} RF net(s) electrically short" + (f" ({', '.join(short)})" if short else "")]
    parts += [f"{r['net']}: {r['reason']}" for r in rows if r["status"] != S.PASS.value]
    return _result(RF_LENGTH_CHECK, status, "; ".join(parts), **details)


# --------------------------------------------------------------------------- impedance, width, length, delay


def _class_width(ir: CircuitIR, cls: NetClass) -> tuple[float | None, str]:
    exact, _, how = controlled_width(ir, float(cls.target_z0_ohm.value))  # type: ignore[union-attr]
    if exact is None:
        return None, how
    w = round_up(exact)
    mfg = ir.pcb.manufacturing if ir.pcb is not None else None
    if mfg is not None and mfg.min_track_width_mm is not None and w < float(mfg.min_track_width_mm.value):
        w = float(mfg.min_track_width_mm.value)
    return w, how


def _entry(track: Track, prefix: str) -> dict[str, str]:
    """The ``key=value`` fields of the track provenance's ``derived_from`` entry starting with ``prefix`` (``rule:`` / ``params:``), or ``{}``."""
    for e in track.provenance.derived_from:
        if e.startswith(prefix):
            return dict(kv.split("=", 1) for kv in e[len(prefix):].split(",") if "=" in kv)
    return {}


def _float(text: str | None) -> float | None:
    try:
        v = float(text) if text is not None else None
    except ValueError:
        return None
    return v if v is not None and math.isfinite(v) else None


def _box_distance(x: float, y: float, box: PadBox) -> float:
    return math.hypot(max(abs(x - box.cx) - box.hw, 0.0), max(abs(y - box.cy) - box.hh, 0.0))


def _escape_segment(track: Track) -> str | None:
    """The words for a track that is one segment of an escape stub the router recorded (module docstring), else ``None``.

    An arc is never one: a stub is recorded as straight segments, and an arc between two of its points is other copper.
    """
    if getattr(track, "mid", None) is not None:
        return None
    esc = escape_entry(track)
    if esc is None:
        return None
    label, width, pts = esc
    if abs(width - float(track.width_mm)) > _TOL:
        return None
    ends = {(round(track.start[0], 6), round(track.start[1], 6)), (round(track.end[0], 6), round(track.end[1], 6))}
    if not any(ends == {(round(a[0], 6), round(a[1], 6)), (round(b[0], 6), round(b[1], 6))} for a, b in zip(pts, pts[1:])):
        return None
    return (f"escape stub of pad {label} recorded by the router (routing.maze {ROUTER_ESCAPE_VERSION}: from the pad to the grid, no wider "
            "than the pad and its neighbours' clearance allow)")


def _neckdown(track: Track, pads: dict[str, list[PadBox]] | None) -> tuple[bool, str]:
    """Whether ``track`` is a neck-down the router recorded - a rule's neck-down or an escape stub (module docstring) -, and the words for
    the row. An arc is sampled on its own centreline (points of the arc, their count from its arc length), never on the chord
    between its ends."""
    escape = _escape_segment(track)
    if escape is not None:
        return True, escape
    rule = _entry(track, "rule:")
    neck, radius = _float(rule.get("neckdown_width")), _float(rule.get("neckdown_radius"))
    if neck is None or radius is None or abs(neck - float(track.width_mm)) > _TOL:
        return False, "not a neck-down the router recorded (its provenance names no neck-down rule at this width)"
    grid = _float(_entry(track, "params:").get("grid")) or 0.0
    reach = radius + grid
    if pads is None:
        return True, f"neck-down recorded by the router (within {radius:g} mm + one {grid:g} mm grid step of a pad; its position not checked: no KiCad library)"
    boxes = [b for bs in pads.values() for b in bs]
    (ax, ay), (bx, by) = track.start, track.end
    mid = getattr(track, "mid", None)
    steps = max(1, int(math.ceil(track_length(track) / max(grid, 0.05))))
    for k in range(steps + 1):
        if mid is None:
            x, y = ax + (bx - ax) * k / steps, ay + (by - ay) * k / steps
        else:
            x, y = arc_point_at(track.start, mid, track.end, k / steps)
        if not any(_box_distance(x, y, b) <= reach + 1e-6 for b in boxes):
            return False, f"its rule names a neck-down, but the track leaves {radius:g} mm + one {grid:g} mm grid step of the net's pads"
    return True, f"neck-down recorded by the router within {radius:g} mm + one {grid:g} mm grid step of the net's pads"


def impedance_result(ir: CircuitIR, si: SIConstraints, cls: NetClass, measures: dict[str, NetMeasure],
                     pads: NetPads | None = None) -> ValidationResult:
    check = f"si.impedance.{cls.name}"
    if cls.target_z0_ohm is None:
        return _result(check, S.NOT_APPLICABLE, f"class {cls.name} states no impedance target")
    nets = _members(ir, si, cls)
    target, tol = float(cls.target_z0_ohm.value), float(cls.z0_tol_rel.value)  # type: ignore[union-attr]
    lo, hi = target * (1.0 - tol), target * (1.0 + tol)
    base = {"target_ohm": target, "tol_rel": tol, "window_ohm": [lo, hi], "nets": nets}
    if not nets:
        return _result(check, S.NOT_APPLICABLE, f"class {cls.name} has no net (nothing declared, nothing promoted)", **base)
    stackup = ir.pcb.stackup if ir.pcb is not None else None
    if stackup is None:
        return _result(check, S.NOT_VERIFIED, f"{NO_STACKUP}: the impedance of {', '.join(nets)} cannot be computed", **base)
    width, how = _class_width(ir, cls)
    rows: list[dict] = []
    necks: list[str] = []
    by_net: dict[str, list[Track]] = {}
    for t in ir.pcb.tracks if ir.pcb is not None else []:
        by_net.setdefault(t.net, []).append(t)
    models: dict[tuple[str, float], tuple[Any, str | None]] = {}
    for net in nets:
        m = measures.get(net)
        if m is None or not m.segments:
            status = S.NOT_APPLICABLE if _pads(ir, net) < 2 else S.NOT_VERIFIED
            rows.append({"net": net, "status": status.value, "reason": "no routed copper" if status is S.NOT_VERIFIED else "fewer than two pads"})
            continue
        # the net's tracks grouped by (layer, width, neck-down or not, the keep-out that clears its plane or None), in segment order
        groups: dict[tuple[str, float, bool, str | None], list[float | str]] = {}
        for seg in m.segments:
            for t in by_net.get(net, []):
                if t.layer != seg.layer or float(t.width_mm) != seg.width_mm:
                    continue
                neck, words = (False, "")
                if width is not None and seg.width_mm < width - _TOL:
                    neck, words = _neckdown(t, None if pads is None else pads.pads.get(net, {}))
                ko = plane_keepout(ir, t) if seg.keepout_mm > 0.0 else None
                g = groups.setdefault((seg.layer, seg.width_mm, neck, ko), [0.0, words])
                g[0] = float(g[0]) + track_length(t)
                if not g[1]:
                    g[1] = words
        for (layer, w, neck, ko), (length, words) in groups.items():
            row: dict[str, Any] = {"net": net, "layer": layer, "width_mm": w, "length_mm": round(float(length), 6)}
            if ko is not None:  # a keep-out clears the plane under this copper: no microstrip there (tools/si/measure.py)
                row.update(status=S.NOT_VERIFIED.value, reason=ko)
                rows.append(row)
                continue
            if neck:
                row.update(status=S.NOT_APPLICABLE.value, reason=f"{words} ({w:g} mm < the controlled {width:g} mm): not judged")
                necks.append(f"{net} {layer} {float(length):.3f} mm at {w:g} mm")
                rows.append(row)
                continue
            if (layer, w) not in models:
                models[(layer, w)] = line_model(stackup, layer, w)
            model, why = models[(layer, w)]
            narrower = f" (narrower than the controlled {width:g} mm and {words})" if width is not None and w < width - _TOL else ""
            if model is None or model.z0_ohm is None:
                row.update(status=S.NOT_VERIFIED.value, reason=undefined_impedance_reason(model, why) + narrower)
            else:
                z = model.z0_ohm
                row.update(z0_ohm=round(z, 4), reference=model.reference, stackup_ids=list(model.ids), notes=list(model.notes))
                if lo - _TOL <= z <= hi + _TOL:
                    row.update(status=S.PASS.value, reason=f"Z0 {z:.2f} ohm within {target:g} ohm +/- {tol:.0%}{narrower}")
                else:
                    row.update(status=S.FAIL.value, reason=f"Z0 {z:.2f} ohm outside {target:g} ohm +/- {tol:.0%} ({lo:.2f}..{hi:.2f}){narrower}", repair="human")
            rows.append(row)
    status = _worst(rows)
    details = {**base, "controlled_width_mm": width, "width_derivation": how, "segments": rows, "neckdowns": necks}
    if status == S.NOT_APPLICABLE:
        return _result(check, status, f"class {cls.name}: no judged segment", **details)
    judged = [r for r in rows if r["status"] == S.PASS.value]
    if status == S.PASS:
        msg = f"{len(judged)} segment(s) of {len(nets)} net(s) within {target:g} ohm +/- {tol:.0%} (calc.tline.microstrip.z0 over the plane)"
    else:
        bad = [f"{r['net']} {r.get('layer', '')} {r.get('width_mm', '')} mm: {r['reason']}" for r in rows if r["status"] in (S.FAIL.value, S.NOT_VERIFIED.value)]
        msg = f"class {cls.name}: " + "; ".join(bad)
    if necks:
        msg += f"; neck-downs not judged: {', '.join(necks)}"
    if status == S.FAIL:
        details["repair"] = "human"
    return _result(check, status, msg, **details)


def width_result(ir: CircuitIR, si: SIConstraints, cls: NetClass) -> ValidationResult | None:
    if cls.min_width_mm is None:
        return None
    check = f"si.width.{cls.name}"
    need = float(cls.min_width_mm.value)
    nets = _members(ir, si, cls)
    tracks = [t for t in (ir.pcb.tracks if ir.pcb is not None else []) if t.net in set(nets)]
    routed = {t.net for t in tracks}
    zoned = {z.net for z in (ir.pcb.zones if ir.pcb is not None else [])}
    # one row per member, like the length / delay / impedance checks: a member that needs copper (two or more pads) and has no
    # track is a requirement nobody measured - NOT_VERIFIED, never NOT_APPLICABLE ("the class states no width" it does state)
    rows: list[dict[str, Any]] = []
    for net in nets:
        if net in routed:
            continue
        if _pads(ir, net) < 2:
            rows.append({"net": net, "status": S.NOT_APPLICABLE.value, "reason": "fewer than two pads"})
        else:
            plane = " (joined through a plane zone, whose current capacity is not judged here)" if net in zoned else ""
            rows.append({"net": net, "status": S.NOT_VERIFIED.value, "reason": f"no routed track{plane}: the minimum width is not judged"})
    unrouted = [r["net"] for r in rows if r["status"] == S.NOT_VERIFIED.value]
    details: dict[str, Any] = {"min_width_mm": need, "source": cls.min_width_mm.provenance.tool or cls.min_width_mm.provenance.kind.value, "nets": nets,
                               "unrouted": rows}
    narrow = sorted({(t.net, t.layer, float(t.width_mm)) for t in tracks if float(t.width_mm) < need - _TOL})
    details["narrow"] = [{"net": n, "layer": layer, "width_mm": w} for n, layer, w in narrow]
    if narrow:
        details["repair"] = "human"
        return _result(check, S.FAIL, f"class {cls.name}: track(s) narrower than the minimum {need:.6g} mm: " + ", ".join(f"{n} {layer} {w:g} mm" for n, layer, w in narrow), **details)
    if unrouted:
        done = f"; {len(tracks)} track(s) of {len(routed)} other net(s) at least {need:.6g} mm wide" if tracks else ""
        return _result(check, S.NOT_VERIFIED, f"class {cls.name}: {', '.join(unrouted)} {'has' if len(unrouted) == 1 else 'have'} no routed track, "
                       f"the minimum width {need:.6g} mm is not judged{done}", **details)
    if not tracks:
        return _result(check, S.NOT_APPLICABLE, f"class {cls.name}: no member net needs copper (fewer than two pads each)", **details)
    return _result(check, S.PASS, f"class {cls.name}: {len(tracks)} track(s) of {len(routed)} net(s) at least {need:.6g} mm wide", **details)


def length_result(ir: CircuitIR, si: SIConstraints, cls: NetClass, measures: dict[str, NetMeasure]) -> ValidationResult:
    check = f"si.length.{cls.name}"
    if cls.max_length_mm is None:
        return _result(check, S.NOT_APPLICABLE, f"class {cls.name} states no length budget")
    budget = float(cls.max_length_mm.value)
    rows: list[dict] = []
    stackup = ir.pcb.stackup if ir.pcb is not None else None
    for net in _declared_nets(ir, si, cls):
        m = measures.get(net)
        if m is None or not m.routed:
            status = S.NOT_APPLICABLE if _pads(ir, net) < 2 else S.NOT_VERIFIED
            rows.append({"net": net, "status": status.value, "reason": "no routed copper" if status is S.NOT_VERIFIED else "fewer than two pads"})
            continue
        if m.vias and stackup is None:
            rows.append({"net": net, "status": S.NOT_VERIFIED.value, "reason": f"{m.vias} via(s): the barrel length needs a stackup ({NO_STACKUP})",
                         "track_length_mm": round(m.track_length_mm, 6)})
            continue
        ok = m.length_mm <= budget + _TOL
        rows.append({"net": net, "status": (S.PASS if ok else S.FAIL).value, "length_mm": round(m.length_mm, 6), "track_length_mm": round(m.track_length_mm, 6),
                     "vias": m.vias, "via_length_mm": round(m.via_length_mm, 6), "budget_mm": budget,
                     "reason": f"{m.length_mm:.3f} mm {'<=' if ok else '>'} {budget:g} mm (tracks + {m.vias} via barrel(s))"})
    status = _worst(rows)
    msg = "; ".join(f"{r['net']}: {r['reason']}" for r in rows) or "no net"
    details: dict[str, Any] = {"budget_mm": budget, "nets": rows}
    if status == S.FAIL:
        details["repair"] = "human"
    return _result(check, status, f"class {cls.name} length budget {budget:g} mm: {msg}", **details)


def delay_result(ir: CircuitIR, si: SIConstraints, cls: NetClass, measures: dict[str, NetMeasure]) -> ValidationResult:
    check = f"si.delay.{cls.name}"
    if cls.max_delay_s is None:
        return _result(check, S.NOT_APPLICABLE, f"class {cls.name} states no delay budget")
    budget = float(cls.max_delay_s.value)
    rows: list[dict] = []
    for net in _declared_nets(ir, si, cls):
        m = measures.get(net)
        if m is None or not m.routed:
            status = S.NOT_APPLICABLE if _pads(ir, net) < 2 else S.NOT_VERIFIED
            rows.append({"net": net, "status": status.value, "reason": "no routed copper" if status is S.NOT_VERIFIED else "fewer than two pads"})
            continue
        d = m.delay_s
        if d is None:
            rows.append({"net": net, "status": S.NOT_VERIFIED.value, "reason": "; ".join(m.problems)})
            continue
        row: dict[str, Any] = {"net": net, "delay_ps": round(d * 1e12, 4), "budget_ps": round(budget * 1e12, 4), "bound": m.bound}
        if d <= budget + 1e-18:
            row.update(status=S.PASS.value, reason=f"{d * 1e12:.2f} ps <= {budget * 1e12:g} ps" + (" (an upper bound: no reference plane)" if m.bound else ""))
        elif m.bound:
            row.update(status=S.NOT_VERIFIED.value, reason=f"the no-plane upper bound {d * 1e12:.2f} ps exceeds {budget * 1e12:g} ps; the true delay needs a reference plane")
        else:
            row.update(status=S.FAIL.value, reason=f"{d * 1e12:.2f} ps > {budget * 1e12:g} ps")
        rows.append(row)
    status = _worst(rows)
    details: dict[str, Any] = {"budget_s": budget, "nets": rows}
    if status == S.FAIL:
        details["repair"] = "human"
    return _result(check, status, f"class {cls.name} delay budget {budget * 1e12:g} ps: " + ("; ".join(f"{r['net']}: {r['reason']}" for r in rows) or "no net"), **details)


def skew_results(ir: CircuitIR, si: SIConstraints, measures: dict[str, NetMeasure]) -> list[ValidationResult]:
    groups: dict[str, tuple[float, list[str]]] = {}
    for cls in si.net_classes:
        if cls.match_group is None or cls.max_skew_s is None:
            continue
        budget, nets = groups.setdefault(cls.match_group, (float(cls.max_skew_s.value), []))
        nets.extend(n for n in _members(ir, si, cls) if n not in nets)
    out: list[ValidationResult] = []
    for group, (budget, nets) in sorted(groups.items()):
        check = f"si.skew.{group}"
        delays: dict[str, float] = {}
        problems: list[str] = []
        for net in nets:
            m = measures.get(net)
            if m is None or not m.routed:
                problems.append(f"{net}: no routed copper")
            elif m.delay_s is None:
                problems.append(f"{net}: " + "; ".join(m.problems))
            elif m.bound:
                problems.append(f"{net}: only an upper bound of its delay is known (no reference plane)")
            else:
                delays[net] = m.delay_s
        details: dict[str, Any] = {"budget_s": budget, "nets": nets, "delays_ps": {k: round(v * 1e12, 4) for k, v in delays.items()}, "problems": problems}
        if problems:
            out.append(_result(check, S.NOT_VERIFIED, f"match group {group}: " + "; ".join(problems), **details))
            continue
        if len(delays) < 2:
            out.append(_result(check, S.NOT_APPLICABLE, f"match group {group}: fewer than two routed nets", **details))
            continue
        skew = max(delays.values()) - min(delays.values())
        details["skew_ps"] = round(skew * 1e12, 4)
        ok = skew <= budget + 1e-18
        if not ok:
            details["repair"] = "human"
        out.append(_result(check, S.PASS if ok else S.FAIL, f"match group {group}: delay spread {skew * 1e12:.3f} ps {'<=' if ok else '>'} {budget * 1e12:g} ps "
                           f"({', '.join(f'{k} {v * 1e12:.2f} ps' for k, v in delays.items())})", **details))
    return out


# --------------------------------------------------------------------------- differential pairs


def _coupled_pieces(ir: CircuitIR, p: str, n: str) -> list[dict]:
    """Parallel P / N segment pairs on one layer whose edge gap is at most :data:`COUPLED_GAP_FACTOR` widths (:func:`ai_eda.tools.routing.coupling.coupled_pieces`)."""
    tracks = ir.pcb.tracks if ir.pcb is not None else []
    return coupled_pieces([t for t in tracks if t.net == p], [t for t in tracks if t.net == n])


def diff_results(ir: CircuitIR, si: SIConstraints, measures: dict[str, NetMeasure]) -> list[ValidationResult]:
    out: list[ValidationResult] = []
    stackup = ir.pcb.stackup if ir.pcb is not None else None
    for cls in si.net_classes:
        for pr in cls.pairs:
            check = f"si.diff.{pr.name}"
            target, tol = float(cls.target_zdiff_ohm.value), float(cls.zdiff_tol_rel.value)  # type: ignore[union-attr]
            lo, hi = target * (1 - tol), target * (1 + tol)
            details: dict[str, Any] = {"target_ohm": target, "tol_rel": tol, "class": cls.name}
            mp, mn = measures.get(pr.p), measures.get(pr.n)
            if mp is None or mn is None or not mp.routed or not mn.routed:
                out.append(_result(check, S.NOT_VERIFIED, f"pair {pr.name}: a net has no routed copper", **details))
                continue
            if stackup is None:
                out.append(_result(check, S.NOT_VERIFIED, f"pair {pr.name}: {NO_STACKUP}", **details))
                continue
            rows: list[dict] = []
            geom, why = line_geometry(stackup, "F.Cu")
            if geom is None:
                out.append(_result(check, S.NOT_VERIFIED, f"pair {pr.name}: {why} - use pcb_layers=4 or add a plane", **details))
                continue
            tracks = ir.pcb.tracks if ir.pcb is not None else []
            coupled, unc_p, unc_n, pieces = uncoupled_lengths([t for t in tracks if t.net == pr.p], [t for t in tracks if t.net == pr.n])
            for x in pieces:
                g, reason = line_geometry(stackup, x["layer"])
                if g is None:
                    rows.append({**x, "status": S.NOT_VERIFIED.value, "reason": f"{reason} - use pcb_layers=4 or add a plane"})
                    continue
                try:
                    z = edge_coupled_microstrip(x["width_mm"], x["gap_mm"], float(g.h.value), float(g.t.value), float(g.er.value)).z_diff
                except (TLineRangeError, ValueError) as e:
                    rows.append({**x, "status": S.NOT_VERIFIED.value, "reason": f"calc.tline.edge_coupled_microstrip refuses: {e}"})
                    continue
                ok = lo - _TOL <= z <= hi + _TOL
                rows.append({**x, "z_diff_ohm": round(z, 4), "status": (S.PASS if ok else S.FAIL).value,
                             "reason": f"Z_diff {z:.2f} ohm {'within' if ok else 'outside'} {target:g} ohm +/- {tol:.0%}"})
            # an arc is never a coupled piece (ai_eda.tools.routing.coupling): its length is uncoupled copper, and the rows say so
            curved = {net: sum(track_length(t) for t in tracks if t.net == net and getattr(t, "mid", None) is not None) for net in (pr.p, pr.n)}
            if not pieces and any(curved.values()):
                rows.append({"status": S.NOT_VERIFIED.value, "reason": (
                    "no straight coupled section, and the pair holds arc track(s) (" + ", ".join(f"{net} {mm:.3f} mm" for net, mm in curved.items() if mm)
                    + "): curved copper is never a coupled piece (the coupled-line formulas cover straight parallel segments), so whether the two "
                    "nets run side by side there is not judged")})
            elif not pieces:
                rows.append({"status": S.FAIL.value, "reason": "no coupled section: the two nets never run side by side"})
            if cls.pair_uncoupled_max_mm is not None:
                budget = float(cls.pair_uncoupled_max_mm.value)
                for net, unc in ((pr.p, unc_p), (pr.n, unc_n)):
                    ok = unc <= budget + 1e-6
                    rows.append({"net": net, "uncoupled_mm": round(unc, 6), "budget_mm": budget, "status": (S.PASS if ok else S.FAIL).value,
                                 "reason": f"{net} uncoupled {unc:.3f} mm {'<=' if ok else '>'} {budget:g} mm"})
                    if curved[net]:
                        rows[-1]["curved_mm"] = round(curved[net], 6)
                        rows[-1]["reason"] += f" ({curved[net]:.3f} mm of it in arc track(s): curved copper is never counted as coupled)"
            if cls.pair_max_skew_s is not None:
                budget_s = float(cls.pair_max_skew_s.value)
                dp, dn = mp.delay_s, mn.delay_s
                if dp is None or dn is None or mp.bound or mn.bound:
                    rows.append({"status": S.NOT_VERIFIED.value, "reason": "intra-pair skew: the delays are unknown or only bounded"})
                else:
                    skew = abs(dp - dn)
                    ok = skew <= budget_s + 1e-18
                    rows.append({"skew_ps": round(skew * 1e12, 4), "budget_ps": budget_s * 1e12, "status": (S.PASS if ok else S.FAIL).value,
                                 "reason": f"intra-pair skew {skew * 1e12:.3f} ps {'<=' if ok else '>'} {budget_s * 1e12:g} ps"})
            status = _worst(rows)
            details.update(rows=rows, coupled_mm=round(coupled, 6))
            if status == S.FAIL:
                details["repair"] = "human"
            out.append(_result(check, status, f"pair {pr.name}: " + "; ".join(r["reason"] for r in rows), **details))
    return out


# --------------------------------------------------------------------------- crosstalk (matched-termination closed forms: rank and explain)

XT_CHECK = "si.crosstalk"
#: tool id / version of the crosstalk rows (their own: the typed-via re-stamp of SI_TOOL never touches them; 0.2: a row whose decks
#: spice.xtalk can build hands off to it)
XT_TOOL = "si.crosstalk"
XT_VERSION = "0.2"
#: the version the crosstalk rows of a board with an arc track carry (:func:`xt_version`): an arc takes part in the neighbour search
#: through its exact box and its chords, and a curved neighbour is named, never computed - a mechanism that acts only on those boards
#: stamps its own; every other board stays XT_VERSION
XT_ARC_VERSION = "0.3"
#: net kinds that are victims and aggressors: a supply / ground net states no driven edge
XT_KINDS = frozenset({NetKind.SIGNAL, NetKind.ANALOG, NetKind.HIGH_SPEED, NetKind.RF})
#: the window: an edge gap of at most this many dielectric heights - the gap calc.tline.edge_coupled_microstrip refuses beyond
XT_G_MAX = COUPLED_G_RANGE[1]
XT_ESTIMATE = "matched-termination estimate, not a verdict"
XT_NON_PARALLEL = "non-parallel neighbour not computed"
#: a neighbour within the window where the victim's track or the neighbour's is an arc (``Track.mid``)
XT_CURVED = "curved neighbour not computed (an arc track, the victim's or the neighbour's: the coupled-line formulas cover straight parallel copper only)"
XT_NO_ASYMMETRIC = "no opened asymmetric coupled-line formula"
XT_NO_BROADSIDE = "no opened broadside formula"
XT_NOT_JUDGED = [
    "pads and vias as aggressors (only tracks are)",
    "power and ground copper beside a victim (no driven edge; a coplanar return is not modelled either)",
    "conductor and dielectric loss, dispersion, the solder mask (calc.tline is quasi-static and lossless)",
    "reflections at the victim's real ends (the forms assume Z0 at both)",
]
_XT_FORMULAS = {
    "modes": "Z_even, Z_odd, e_eff_even, e_eff_odd: calc.tline.edge_coupled_microstrip (Kirschning & Jansen 1984, static) at the victim width, gap s, h, t, er",
    "kb": "Kb = (Z_even - Z_odd) / (Z_even + Z_odd + 2 Z_even Z_odd / Z_term), Z_term = sqrt(Z_even Z_odd) (Simonovich, SIJ 2022, Eq. 8 and Eq. 2)",
    "l_sat": "L_sat = t_r / (2 t_pd_even), t_pd_even = sqrt(e_eff_even) / c0 (Simonovich Eq. 4; the slower mode is a choice: the shorter L_sat)",
    "next": "NEXT_rel = Kb min(1, L / L_sat) (Simonovich Eq. 6)",
    "fext": "FEXT_rel = -min(L |sqrt(e_eff_even) - sqrt(e_eff_odd)| / (2 c0 t_r), 1/2) (Polar AP8164 velocity form with LaMeres slides 21/31's 1/2; "
            "the 1/2 cap from Macnica 127093 Fig. 4)",
    "sum": "matched_estimate_rel = sum over the computed pieces of |NEXT_rel| + |FEXT_rel| per unit launched aggressor step (superposition, LaMeres slide 2): "
           "a ranking number",
    "t_r": "the aggressor class's edge (tools.si.driver.driver_value t_rise_s) used as the 0-100 % linear ramp, as spice.si's PULSE uses it",
}


def xt_version(ir: CircuitIR) -> str:
    """The version ``ir``'s ``si.crosstalk`` rows are stamped with: :data:`XT_ARC_VERSION` when its board holds an arc track
    (``Track.mid``), else :data:`XT_VERSION` - a board without one is searched and stamped exactly as before."""
    pcb = ir.pcb
    return XT_ARC_VERSION if pcb is not None and any(getattr(t, "mid", None) is not None for t in pcb.tracks) else XT_VERSION


def _xt_result(version: str, status: S, message: str, **details: Any) -> ValidationResult:
    return ValidationResult(check_id=XT_CHECK, status=status, message=f"{message} ({NOT_DRC})", tool=XT_TOOL, tool_version=version,
                            details={"kind": KIND, **details})


def _g6(x: float) -> float:
    return float(f"{x:.6g}")


@dataclass(frozen=True)
class _Run:
    """A straight run of one net's copper: collinear tracks of one width on one layer that touch or overlap, merged (``tracks`` their
    indices) - or one arc track by itself, with its ``mid`` point (``None`` for a straight run): shaped like a track, so the neighbour
    search and the keep-out test read it as the arc it is."""

    net: str
    layer: str
    start: tuple[float, float]
    end: tuple[float, float]
    width_mm: float
    tracks: tuple[int, ...]
    mid: tuple[float, float] | None = None


def _straight_runs(tracks: list[Track], indices: list[int]) -> list[_Run]:
    """The tracks ``indices`` (of ``tracks``) merged into straight runs, in order of first appearance (a router may split a straight run).

    Two tracks join when they share the net, layer and width, are parallel
    (the :func:`~ai_eda.tools.routing.coupling.coupled_pieces` test), lie on
    one line (within 1e-6 mm) and their extents touch or overlap (within
    1e-6 mm). Zero-length tracks are dropped. An arc track is never part of
    a straight run: it is a run of its own (:attr:`_Run.mid`), whatever the
    line through its two ends, so it ends the straight runs beside it.
    """
    groups: dict[tuple[str, str, float], list[list]] = {}
    order: list[list] = []
    for i in indices:
        t = tracks[i]
        if getattr(t, "mid", None) is not None:
            order.append([None, None, None, None, None, tuple(t.start), tuple(t.end), [i]])  # listed, in no group: nothing merges with it
            continue
        (x0, y0), (x1, y1) = t.start, t.end
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length <= _TOL:
            continue
        if dx < 0 or (dx == 0 and dy < 0):
            x0, y0, x1, y1, dx, dy = x1, y1, x0, y0, -dx, -dy
        ux, uy = dx / length, dy / length
        # direction, line offset, extent along the line and its end points, sources
        seg = [ux, uy, ux * y0 - uy * x0, ux * x0 + uy * y0, ux * x1 + uy * y1, (x0, y0), (x1, y1), [i]]
        runs = groups.setdefault((t.net, t.layer, float(t.width_mm)), [])
        merged = True
        while merged:
            merged = False
            for r in runs:
                if abs(r[0] * seg[1] - r[1] * seg[0]) <= 1e-6 and abs(r[2] - seg[2]) <= 1e-6 and seg[3] <= r[4] + 1e-6 and r[3] <= seg[4] + 1e-6:
                    runs.remove(r)
                    order.remove(r)
                    lo, hi = (r if r[3] <= seg[3] else seg), (r if r[4] >= seg[4] else seg)
                    seg = [r[0], r[1], r[2], lo[3], hi[4], lo[5], hi[6], sorted(r[7] + seg[7])]
                    merged = True
                    break
        runs.append(seg)
        order.append(seg)
    out: list[_Run] = []
    for seg in sorted(order, key=lambda s: s[7][0]):
        t = tracks[seg[7][0]]
        start, end = (tuple(t.start), tuple(t.end)) if len(seg[7]) == 1 else (seg[5], seg[6])
        out.append(_Run(net=t.net, layer=t.layer, start=start, end=end, width_mm=float(t.width_mm), tracks=tuple(seg[7]), mid=getattr(t, "mid", None)))
    return out


def _xt_layers(stackup) -> tuple[dict[str, float], dict[str, dict], dict[str, dict[str, float]], dict[str, str]]:
    """``(window h per layer, how each h was taken, broadside neighbours, unsearched layers -> why)`` of a stack (:func:`crosstalk_result`)."""
    window: dict[str, float] = {}
    info: dict[str, dict] = {}
    unsearched: dict[str, str] = {}
    names = stackup.copper_names()
    for i, name in enumerate(names):
        geom, why = line_geometry(stackup, name)
        if geom is not None:
            window[name] = float(geom.h.value)
            info[name] = {"h_mm": window[name], "h_id": geom.h_id, "how": f"the microstrip dielectric over {geom.reference_layer} (calc.tline.line_geometry)"}
            continue
        if stackup.copper[i].plane_net is not None:
            unsearched[name] = str(why)
            continue
        near = [k for k in (i - 1, i) if 0 <= k < len(stackup.dielectrics)]
        if not near:
            unsearched[name] = str(why)
            continue
        k = min(near, key=lambda k: (float(stackup.dielectrics[k].thickness_mm.value), k))
        window[name] = float(stackup.dielectrics[k].thickness_mm.value)
        info[name] = {"h_mm": window[name], "h_id": f"pcb.stackup.dielectrics[{k}].thickness_mm",
                      "how": f"no microstrip geometry ({why}): the window uses the thinner dielectric next to the layer - a choice, so its neighbours are named, "
                             "not computed"}
    adjacent: dict[str, dict[str, float]] = {}
    for i in range(len(names) - 1):
        if stackup.copper[i].plane_net is None and stackup.copper[i + 1].plane_net is None:
            d = float(stackup.dielectrics[i].thickness_mm.value)
            adjacent.setdefault(names[i], {})[names[i + 1]] = d
            adjacent.setdefault(names[i + 1], {})[names[i]] = d
    return window, info, adjacent, unsearched


@dataclass
class CrosstalkScan:
    """:func:`crosstalk_scan`'s result, the victims' computed pieces (with their end points on both tracks) and the deck plan of each
    victim that hands off to ``spice.xtalk``."""

    result: ValidationResult
    pieces: dict[str, list[XtPiece]]
    plans: dict[str, VictimPlan]


def crosstalk_result(ir: CircuitIR, si: SIConstraints, measures: dict[str, NetMeasure] | None = None, pads: NetPads | None = None) -> ValidationResult:
    """``si.crosstalk`` (:func:`crosstalk_scan`)."""
    return crosstalk_scan(ir, si, measures, pads).result


def crosstalk_scan(ir: CircuitIR, si: SIConstraints, measures: dict[str, NetMeasure] | None = None, pads: NetPads | None = None) -> CrosstalkScan:
    """``si.crosstalk``: one row per IR net - the other nets' copper beside it and the matched-termination closed forms of each parallel run.

    Victims and aggressors are the IR nets of :data:`XT_KINDS` (a supply or
    ground net states no edge: a NOT_APPLICABLE row, never an aggressor).
    Each net's touching collinear tracks of one width are first merged into
    straight runs (:func:`_straight_runs`: a router may split a straight run,
    and the split is no neighbour - a choice of this check; an arc track is a
    run of its own). The neighbour
    search (:class:`ai_eda.tools.routing.coupling.NeighbourIndex`)
    looks on every searched layer for another net's copper within an edge
    gap of :data:`XT_G_MAX` h, h the layer's microstrip dielectric
    (:func:`~ai_eda.tools.calc.tline.line_geometry`; a layer without that
    geometry uses the thinner dielectric next to it - a choice, its
    neighbours are only named). Each exactly parallel run of equal width over
    a plane is computed with :mod:`ai_eda.tools.calc.xtalk` (``details["formulas"]``):
    Kb, L_sat, NEXT_rel and FEXT_rel per piece, their absolute sum per victim
    ``matched_estimate_rel`` - a ranking number. What no opened formula
    covers is named, never estimated: copper on a layer with no microstrip
    geometry (the ``line_geometry`` text), a plane a keep-out clears
    (``plane_keepout``), unequal widths (:data:`XT_NO_ASYMMETRIC`), a gap the
    coupled-line formula refuses (its ``TLineRangeError``), a broadside
    overlap on an adjacent signal layer with no plane between
    (:data:`XT_NO_BROADSIDE`), a non-parallel neighbour (:data:`XT_NON_PARALLEL`),
    a curved one (:data:`XT_CURVED`: the victim's track or the neighbour's is
    an arc - found through the arc's exact box and its chords, named with the
    smallest gap the copper can have, one piece per pair of tracks, never fed
    to a closed form),
    another net's zone, touching copper, a missing aggressor edge (the
    ``driver_value`` text). Row status, first match wins: NOT_APPLICABLE for a
    supply / ground net, a net without a routed track, no other net's copper
    within the window (``beyond_window`` counts the other nets on its layers);
    NOT_VERIFIED "no stackup"; NOT_VERIFIED naming every item not computed;
    otherwise - every neighbour computed, but every opened closed form
    assumes Z0 at both victim ends and an ai_eda net is driven through
    ``r_drive_ohm`` into ``c_load_f`` - NOT_APPLICABLE "judged by
    spice.xtalk.<victim>" when :func:`ai_eda.tools.spice.xtalk_check.plan_victim`
    can build its coupled-line decks (``judged_by`` names that check, as
    ``si.critical_length`` hands a long net to ``spice.si``), else
    NOT_VERIFIED "the matched-termination closed forms do not bound a victim
    driven through R ohm into C F; no coupled-line deck: <why>" (more than two
    pads, an unmodelled part, no extracted path - inner-layer copper, a typed
    via, no KiCad library -, a piece off the path). The check never says PASS
    or FAIL. The result is the worst row (no row: NOT_APPLICABLE).
    ``measures`` gives the nets' pad-to-pad paths (measured without a KiCad
    library when ``None``: then no path, no hand-off); ``pads`` is not read.
    Deterministic: IR net order, then track order.
    """
    from ai_eda.tools.spice.xtalk_check import XtPiece, plan_victim

    if measures is None:
        measures = measure_nets(ir)
    pcb = ir.pcb
    version = xt_version(ir)
    tracks = list(pcb.tracks) if pcb is not None else []
    stackup = pcb.stackup if pcb is not None else None
    kinds = {n.name: n.kind for n in ir.nets}
    by_net: dict[str, list[int]] = {}
    for i, t in enumerate(tracks):
        by_net.setdefault(t.net, []).append(i)
    aggr = [i for i, t in enumerate(tracks) if kinds.get(t.net) in XT_KINDS]
    base: dict[str, Any] = {"window": f"edge gap s <= {XT_G_MAX:g} h (the gap calc.tline.edge_coupled_microstrip refuses beyond: COUPLED_G_RANGE)",
                            "estimate": XT_ESTIMATE, "formulas": dict(_XT_FORMULAS), "not_judged": list(XT_NOT_JUDGED),
                            "swing": "every relative number is per unit launched aggressor step; the aggressor swing is taken equal to the victim swing"}
    window: dict[str, float] = {}
    adjacent: dict[str, dict[str, float]] = {}
    unsearched: dict[str, str] = {}
    if stackup is not None:
        window, info, adjacent, unsearched = _xt_layers(stackup)
        base["layers"] = info
        base["broadside_pairs"] = sorted({tuple(sorted((a, b))) for a, row in adjacent.items() for b in row})
    zones = [z for z in (pcb.zones if pcb is not None else []) if kinds.get(z.net) in XT_KINDS and z.layer not in unsearched]
    runs = _straight_runs(tracks, aggr)
    runs_of: dict[str, list[int]] = {}
    for k, run in enumerate(runs):
        runs_of.setdefault(run.net, []).append(k)
    index = NeighbourIndex(runs, window, g_max=XT_G_MAX, adjacent=adjacent, zones=zones)
    nets_on: dict[str, set[str]] = {}
    for i in aggr:
        nets_on.setdefault(tracks[i].layer, set()).add(tracks[i].net)
    edges: dict[str, tuple[Any, str]] = {}
    coupled: dict[tuple[str, float, float], Any] = {}
    keepouts: dict[int, str | None] = {}

    def edge(net: str) -> tuple[Any, str]:
        if net not in edges:
            cls = si.class_of(net)
            if cls is None:
                edges[net] = (None, f"{net} is in no net class (no default class), so its edge is not stated")
            else:
                edges[net] = driver_value(ir, cls, "t_rise_s", net)
        return edges[net]

    def keepout(k: int) -> str | None:
        if k not in keepouts:
            keepouts[k] = plane_keepout(ir, runs[k])
        return keepouts[k]

    rows: list[dict] = []
    pieces_of: dict[str, list[XtPiece]] = {}
    plans: dict[str, VictimPlan] = {}
    for n in ir.nets:
        cls = si.class_of(n.name)
        row: dict[str, Any] = {"net": n.name, "class": None if cls is None else cls.name, "judged_by": None, "matched_estimate_rel": None,
                               "not_computed": [], "beyond_window": 0, "aggressors": []}
        rows.append(row)
        mine = by_net.get(n.name, [])
        if n.kind not in XT_KINDS:
            row.update(status=S.NOT_APPLICABLE.value, reason=f"a {n.kind.value} net carries no driven edge: neither victim nor aggressor")
            continue
        if not mine:
            row.update(status=S.NOT_APPLICABLE.value, reason="no routed track")
            continue
        if stackup is None:
            row.update(status=S.NOT_VERIFIED.value, reason=f"{NO_STACKUP}: the window (s <= {XT_G_MAX:g} h) and the coupled-line values need the dielectric")
            continue
        mine_runs = runs_of.get(n.name, [])
        victim = [runs[k] for k in mine_runs]
        items: dict[tuple[str, str, str], dict[str, Any]] = {}

        def name(kind: str, layer: str, reason: str, net: str, gap: float | None = None, length: float | None = None) -> None:
            it = items.setdefault((kind, layer, reason), {"kind": kind, "layer": layer, "reason": reason, "nets": [], "pieces": 0})
            if net not in it["nets"]:
                it["nets"].append(net)
            it["pieces"] += 1
            if gap is not None:
                it["min_gap_mm"] = round(min(gap, it.get("min_gap_mm", gap)), 6)
            if length is not None:
                it["length_mm"] = round(it.get("length_mm", 0.0) + length, 6)

        for layer in dict.fromkeys(tracks[i].layer for i in mine):
            if layer not in window and layer in unsearched:
                name("victim_copper", layer, f"the victim's own copper on {layer} is not searched: {unsearched[layer]}", n.name)
            elif layer not in window:
                name("victim_copper", layer, f"the victim's own copper on {layer} is not searched: layer {layer!r} is not in the stackup", n.name)
        near: set[str] = set()
        total = 0.0
        computed = 0
        for rec in index.runs(victim):
            a_net, kind, layer = rec["net"], rec["kind"], rec["layer"]
            near.add(a_net)
            if kind == "non_parallel" and rec.get("curved"):
                name(kind, layer, XT_CURVED, a_net, rec["gap_mm"])
                continue
            if kind == "non_parallel":
                name(kind, layer, XT_NON_PARALLEL + (" (parallel, no projected overlap)" if rec["parallel"] else ""), a_net, rec["gap_mm"])
                continue
            if kind == "contact":
                name(kind, layer, "copper of another net touches or crosses the victim (edge gap <= 0): a short for DRC to judge, not a coupled line", a_net)
                continue
            if kind == "zone":
                name(kind, layer, "another net's zone on a signal layer inside the window: not computed", a_net, rec["gap_mm"])
                continue
            if kind == "broadside":
                name(kind, layer, f"{XT_NO_BROADSIDE} ({rec['victim_layer']} over {layer} with no plane between, dielectric {rec['h_mm']:g} mm)", a_net,
                     rec["lateral_gap_mm"], rec["length_mm"])
                continue
            vt, at = victim[rec["victim_index"]], runs[rec["aggressor_index"]]
            length, gap = float(rec["length_mm"]), float(rec["gap_mm"])
            stop: list[str] = []
            geom, why = line_geometry(stackup, layer)
            if geom is None:
                stop.append(str(why))
            for k in (mine_runs[rec["victim_index"]], rec["aggressor_index"]):
                ko = keepout(k)
                if ko is not None:
                    stop.append(ko)
            wv, wa = float(vt.width_mm), float(at.width_mm)
            if abs(wv - wa) > 1e-6:
                stop.append(f"{XT_NO_ASYMMETRIC} (victim {wv:g} mm, aggressor {wa:g} mm; Kirschning-Jansen covers equal widths only)")
            r = None
            if not stop and geom is not None:
                key = (layer, wv, gap)
                if key not in coupled:
                    try:
                        coupled[key] = edge_coupled_microstrip(wv, gap, float(geom.h.value), float(geom.t.value), float(geom.er.value))
                    except (TLineRangeError, ValueError) as e:
                        coupled[key] = f"calc.tline.edge_coupled_microstrip refuses: {e}"
                r = coupled[key]
                if isinstance(r, str):
                    stop.append(r)
                    r = None
            t_r, missing = edge(a_net)
            if t_r is None:  # one item per missing value; the item lists the aggressor nets
                stop.append(f"an aggressor's edge is missing: {missing}")
            for reason in stop:
                name("parallel", layer, reason, a_net, gap, length)
            if r is None:
                continue
            assert geom is not None
            kb = kb_eq8(r.z_even, r.z_odd, math.sqrt(r.z_even * r.z_odd))
            entry: dict[str, Any] = {"net": a_net, "layer": layer, "length_mm": round(length, 6), "width_mm": wv, "gap_mm": round(gap, 6),
                                     "s_over_h": round(gap / float(geom.h.value), 6), "z_even_ohm": round(r.z_even, 4), "z_odd_ohm": round(r.z_odd, 4),
                                     "e_eff_even": round(r.e_eff_even, 6), "e_eff_odd": round(r.e_eff_odd, 6), "kb": _g6(kb),
                                     "l_sat_mm": None, "next_rel": None, "fext_rel": None, "t_r_s": None, "t_r_source": missing, "ids": list(geom.ids)}
            if t_r is not None:
                ls = l_sat_mm(t_r.value, propagation_delay(r.e_eff_even))
                nx, fx = next_rel(kb, length, ls), fext_rel(length, r.e_eff_even, r.e_eff_odd, t_r.value)
                entry.update(l_sat_mm=round(ls, 6), next_rel=_g6(nx), fext_rel=_g6(fx), t_r_s=t_r.value, t_r_source=t_r.source)
                total += abs(nx) + abs(fx)
                computed += 1
                vs, as_ = rec["victim_span"], rec["aggressor_span"]
                pieces_of.setdefault(n.name, []).append(XtPiece(
                    aggressor=a_net, layer=layer, width_mm=wv, gap_mm=gap, length_mm=length, victim_span=(tuple(vs[0]), tuple(vs[1])),
                    aggressor_span=(tuple(as_[0]), tuple(as_[1])), z_even=r.z_even, z_odd=r.z_odd, e_eff_even=r.e_eff_even, e_eff_odd=r.e_eff_odd))
            row["aggressors"].append(entry)
        row["not_computed"] = list(items.values())
        row["beyond_window"] = len(set().union(*(nets_on.get(t.layer, set()) for t in victim)) - near - {n.name})
        if computed:
            row["matched_estimate_rel"] = _g6(total)
        if not near and not items:
            row.update(status=S.NOT_APPLICABLE.value, reason=f"no other net's copper within s <= {XT_G_MAX:g} h of its tracks"
                       + (f" ({row['beyond_window']} other net(s) on its layers lie beyond the window)" if row["beyond_window"] else ""))
        elif items:
            row.update(status=S.NOT_VERIFIED.value, reason="; ".join(dict.fromkeys(it["reason"] for it in items.values())))
        else:
            rv, why_r = driver_value(ir, cls, "r_drive_ohm", n.name) if cls is not None else (None, f"{n.name} is in no net class")
            cv, why_c = driver_value(ir, cls, "c_load_f", n.name) if cls is not None else (None, f"{n.name} is in no net class")
            if rv is not None and cv is not None:
                ends = f"do not bound a victim driven through {rv.value:g} ohm into {cv.value:g} F"
            else:
                ends = "assume Z0 at both victim ends, and its termination is not stated (" + "; ".join(w for w in (why_r, why_c) if w) + ")"
            plan = plan_victim(ir, n.name, pieces_of.get(n.name, []), measures)
            if isinstance(plan, str):
                row.update(status=S.NOT_VERIFIED.value, reason=f"the matched-termination closed forms {ends}; no coupled-line deck: {plan}")
            else:
                plans[n.name] = plan
                row.update(status=S.NOT_APPLICABLE.value, judged_by=f"spice.xtalk.{n.name}",
                           reason=f"judged by spice.xtalk.{n.name} ({plan.decks} coupled-line deck(s)), not by this rule: the matched-termination closed forms {ends}")
    status = _worst(rows)
    details = {**base, "rows": rows}
    nv = [r for r in rows if r["status"] == S.NOT_VERIFIED.value]
    handed = [r["net"] for r in rows if r["judged_by"] is not None]
    if status == S.NOT_APPLICABLE:
        if handed:
            return CrosstalkScan(_xt_result(version, status, f"every victim with a neighbour is judged by spice.xtalk: {', '.join(handed)}", **details), pieces_of, plans)
        judged = sum(1 for r in rows if r["reason"].startswith("no other net's copper"))
        return CrosstalkScan(_xt_result(version, status, f"no victim has another net's copper within s <= {XT_G_MAX:g} h ({judged} routed signal net(s) searched)",
                                        **details), pieces_of, plans)
    estimates = [r for r in rows if r["matched_estimate_rel"] is not None]
    parts = [f"{len(nv)} victim(s) NOT_VERIFIED - the closed forms rank and explain, they never PASS or FAIL a net"]
    if stackup is None:
        parts.append(f"{NO_STACKUP}: no window and no coupled-line value")
    complete = [r["net"] for r in nv if not r["not_computed"] and r["aggressors"]]
    if complete:
        parts.append(f"{len(complete)} with every neighbour computed but driven through R_drive into C_load, not Z0, and no coupled-line deck (named per row)")
    partial = [r["net"] for r in nv if r["not_computed"]]
    if partial:
        parts.append(f"{len(partial)} with neighbours not computed (named per row)")
    if handed:
        parts.append(f"{len(handed)} judged by spice.xtalk: {', '.join(handed)}")
    if estimates:
        top = max(estimates, key=lambda r: (r["matched_estimate_rel"], r["net"]))
        parts.append(f"largest {XT_ESTIMATE}: {top['matched_estimate_rel']:.3%} of the aggressor step on {top['net']}")
    others = [r for r in rows if r["status"] == S.NOT_APPLICABLE.value and r["judged_by"] is None]
    if others:
        parts.append(f"{len(others)} net(s) not applicable")
    return CrosstalkScan(_xt_result(version, status, "; ".join(parts), **details), pieces_of, plans)


# --------------------------------------------------------------------------- timing paths


def _term(ir: CircuitIR, path: TimingPath, term: str) -> tuple[float | None, str]:
    """``(value s, source)`` of a timing term, or ``(None, what is missing)``."""
    own: Traced | None = getattr(path, term)
    if own is not None:
        return float(own.value), f"si.timing_paths[{path.name}].{term} ({own.provenance.kind.value})"
    ref = path.terms_from.get(term)
    if ref is None:
        return None, f"{term} (neither given nor named in terms_from)"
    m = FACT_REF_RE.match(ref)
    assert m is not None
    comp = ir.component(m["ref"])
    fact = comp.electrical.get(m["key"]) if comp is not None else None
    if comp is None:
        return None, f"{ref} (no component {m['ref']} in the IR: nothing can ground it here)"
    if fact is None or not fact.provenance.is_authoritative or fact.unit != TIMING_TERMS[term] or isinstance(fact.value, bool) or not isinstance(fact.value, (int, float)):
        if comp.symbol is not None and comp.symbol.library.startswith(CONNECTOR_LIBRARY_PREFIX):
            return None, (f"{ref} (datasheet fact key {m['key']} of {m['ref']} not grounded: {m['ref']} is a connector and the part that launches the edge "
                          f"is off-board, behind it - ground the fact on that part's own document: --datasheet-url {m['ref']}=<its document> with a datasheet facts file)")
        return None, f"{ref} (datasheet fact key {m['key']} of {m['ref']} not grounded)"
    return float(fact.value), f"{ref} (grounded datasheet fact)"


def timing_results(ir: CircuitIR, si: SIConstraints, measures: dict[str, NetMeasure]) -> list[ValidationResult]:
    out: list[ValidationResult] = []
    for path in si.timing_paths:
        check = f"si.timing.{path.name}"
        missing: list[str] = []
        sources: dict[str, str] = {}
        values: dict[str, float] = {}
        for term in TIMING_TERMS:
            v, src = _term(ir, path, term)
            if v is None:
                missing.append(src)
            else:
                values[term], sources[term] = v, src
        if path.f_clk_hz is None:
            missing.append(f"si.timing_paths[{path.name}].f_clk_hz")
        if path.capture_fraction is None:
            missing.append(f"si.timing_paths[{path.name}].capture_fraction")
        flights: dict[str, float] = {}
        for net in [path.clock_net, *path.data_nets]:
            m = measures.get(net)
            if m is None or not m.routed:
                missing.append(f"routed copper of {net}")
            elif m.delay_s is None:
                missing.append(f"the delay of {net} ({'; '.join(m.problems)})")
            else:
                flights[net] = m.delay_s
        details: dict[str, Any] = {"clock_net": path.clock_net, "data_nets": list(path.data_nets), "terms": sources, "missing": missing,
                                   "flight_upper_ps": {k: round(v * 1e12, 4) for k, v in flights.items()},
                                   "flight_model": "each flight time in [0, the net's whole routed delay] (the launching-to-capturing pin path is not identified)"}
        if missing:
            out.append(_result(check, S.NOT_VERIFIED, f"timing path {path.name}: missing " + ", ".join(missing), **details))
            continue
        period = 1.0 / float(path.f_clk_hz.value)  # type: ignore[union-attr]
        t_lc = float(path.capture_fraction.value) * period  # type: ignore[union-attr]
        fc = flights[path.clock_net]
        rows: list[dict] = []
        for data in path.data_nets:
            fd = flights[data]
            setup_worst = t_lc + 0.0 - values["t_co_max_s"] - fd - values["t_su_min_s"]
            setup_best = t_lc + fc - values["t_co_max_s"] - 0.0 - values["t_su_min_s"]
            hold_worst = (period - t_lc) + values["t_co_min_s"] + 0.0 - fc - values["t_h_min_s"]
            hold_best = (period - t_lc) + values["t_co_min_s"] + fd - 0.0 - values["t_h_min_s"]
            row: dict[str, Any] = {"data_net": data, "setup_margin_ns": [round(setup_worst * 1e9, 6), round(setup_best * 1e9, 6)],
                                   "hold_margin_ns": [round(hold_worst * 1e9, 6), round(hold_best * 1e9, 6)]}
            if setup_worst >= 0 and hold_worst >= 0:
                row["status"] = S.PASS.value
            elif setup_best < 0 or hold_best < 0:
                row["status"] = S.FAIL.value
            else:
                row["status"] = S.NOT_VERIFIED.value
            rows.append(row)
        status = _worst(rows)
        details.update(period_s=period, launch_to_capture_s=t_lc, data=rows)
        if status == S.FAIL:
            details["repair"] = "human"
        out.append(_result(check, status, f"timing path {path.name}: T = {period * 1e9:.3f} ns, " + "; ".join(
            f"{r['data_net']} setup margin {r['setup_margin_ns'][0]:.3f}..{r['setup_margin_ns'][1]:.3f} ns, hold margin {r['hold_margin_ns'][0]:.3f}..{r['hold_margin_ns'][1]:.3f} ns"
            for r in rows), **details))
    return out


# --------------------------------------------------------------------------- the validator


def si_results(ir: CircuitIR, library: KicadLibrary | None = None) -> list[ValidationResult]:
    """Every ``si.*`` result for ``ir`` (empty when ``ir.si`` is not set); ``library`` gives the pad boxes (pad-to-pad paths, neck-down positions)."""
    si = ir.si
    if si is None:
        return []
    measures = measure_nets(ir, library=library)
    pads = net_pads(ir, library) if library is not None and ir.pcb is not None else None
    out = [critical_result(ir, measures)]
    if rf_nets(ir):
        out.append(rf_length_result(ir, measures))
    for cls in si.net_classes:
        out.append(impedance_result(ir, si, cls, measures, pads))
        w = width_result(ir, si, cls)
        if w is not None:
            out.append(w)
        out.append(length_result(ir, si, cls, measures))
        out.append(delay_result(ir, si, cls, measures))
    out.extend(skew_results(ir, si, measures))
    out.extend(diff_results(ir, si, measures))
    out.append(crosstalk_result(ir, si, measures, pads))
    out.extend(timing_results(ir, si, measures))
    version = si_version(ir)
    if version != SI_VERSION:  # a board with an arc track or a blind / buried / micro via is measured as such: its results say so (every other board: as before)
        out = [r.model_copy(update={"tool_version": version}) if r.tool == SI_TOOL else r for r in out]
    return out


class SIValidator(Validator):
    id = SI_TOOL
    description = "routed copper against the SI net classes and timing paths: impedance, width, length, delay, skew, pairs, critical length, timing (IR geometry + calculators, not DRC)"

    def applies_to(self, ir: CircuitIR) -> bool:
        return ir.si is not None

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        if ir.pcb is None or not ir.pcb.placements:
            return [_result(CRITICAL_CHECK, S.NOT_APPLICABLE, "no placed board: nothing routed to judge")]
        library = ctx.tools.get("kicad_library")
        return si_results(ir, library if isinstance(library, KicadLibrary) else None)


default_registry.register(SIValidator())

__all__ = [
    "CRITICAL_CHECK",
    "SHORT_TEXT",
    "SIValidator",
    "XT_ARC_VERSION",
    "XT_CHECK",
    "XT_TOOL",
    "XT_VERSION",
    "CrosstalkScan",
    "critical_result",
    "crosstalk_result",
    "crosstalk_scan",
    "delay_result",
    "diff_results",
    "impedance_result",
    "length_result",
    "rf_length_result",
    "si_results",
    "skew_results",
    "timing_results",
    "width_result",
    "xt_version",
]
