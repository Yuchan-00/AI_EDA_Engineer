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

The validator applies only to a design that states SI constraints
(``ir.si`` is set): an IR without them gets no result, so older designs keep
their verdicts. Results (tool :data:`~ai_eda.tools.si.measure.SI_TOOL`):

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
  grid step of one of the net's pads. Every other narrower track is judged
  at its own width (a promoted net the router left at the board width
  FAILs). NOT_APPLICABLE for a class without a target or without nets.
* ``si.width.<class>`` - every track of the class's nets at least
  ``min_width_mm`` (a class with one).
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
  P / N delay difference against ``pair_max_skew_s``.
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
from typing import Any

from ai_eda.ir import CircuitIR, NetClass, SIConstraints, TimingPath, Track, Traced, ValidationResult, ValidationStatus, worst_status
from ai_eda.ir.si import FACT_REF_RE, TIMING_TERMS
from ai_eda.tools.calc.tline import NO_STACKUP, TLineRangeError, edge_coupled_microstrip, line_geometry
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.routing.coupling import COUPLED_GAP_FACTOR, coupled_pieces, uncoupled_lengths
from ai_eda.tools.si.measure import NOT_DRC, SI_TOOL, SI_VERSION, NetMeasure, line_model, measure_nets, undefined_impedance_reason
from ai_eda.tools.si.paths import NetPads, PadBox, net_pads
from ai_eda.tools.si.promote import critical_rows
from ai_eda.tools.si.rf import RF_IMPEDANCE_CHECK, RF_LENGTH_CHECK, format_hz, rf_length_rows, rf_nets
from ai_eda.tools.si.rules import controlled_width, round_up
from ai_eda.validation.base import ValidationContext, Validator
from ai_eda.validation.registry import default_registry

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
            row["status"] = S.NOT_VERIFIED.value
            row["reason"] = (f"electrically long ({r.length_mm:.3f} mm > l_crit {r.l_crit_mm:.3f} mm by the upper bound of t_pd{where}): needs impedance control, "
                             "but impedance is undefined without a reference plane - use pcb_layers=4 or add a plane")
            uncontrolled.append(r.net)
        else:
            row["status"] = S.NOT_APPLICABLE.value
            row["reason"] = f"electrically long ({r.length_mm:.3f} mm > l_crit {r.l_crit_mm:.3f} mm{where}): over a plane, judged by spice.si.{r.net}, not by this rule"
            long_spice.append(r.net)
        rows.append(row)
    status = _worst(rows)
    short = [r["net"] for r in rows if r["status"] == S.PASS.value]
    details = {"nets": rows, "long_over_plane": long_spice, "long_without_plane": uncontrolled, "possibly_long": possibly,
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
    if possibly:
        parts.append(f"{len(possibly)} net(s) possibly long (the whole copper exceeds l_crit, no pad-to-pad path extracted; not judged): {', '.join(possibly)}")
    other = [f"{r['net']}: {r['reason']}" for r in rows if r["status"] == S.NOT_VERIFIED.value and r["net"] not in uncontrolled and r["net"] not in possibly]
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


def _neckdown(track: Track, pads: dict[str, list[PadBox]] | None) -> tuple[bool, str]:
    """Whether ``track`` is a neck-down the router recorded (module docstring), and the words for the row."""
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
    steps = max(1, int(math.ceil(math.hypot(bx - ax, by - ay) / max(grid, 0.05))))
    for k in range(steps + 1):
        x, y = ax + (bx - ax) * k / steps, ay + (by - ay) * k / steps
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
        # the net's tracks grouped by (layer, width, neck-down or not), in segment order
        groups: dict[tuple[str, float, bool], list[float | str]] = {}
        for seg in m.segments:
            for t in by_net.get(net, []):
                if t.layer != seg.layer or float(t.width_mm) != seg.width_mm:
                    continue
                neck, words = (False, "")
                if width is not None and seg.width_mm < width - _TOL:
                    neck, words = _neckdown(t, None if pads is None else pads.pads.get(net, {}))
                g = groups.setdefault((seg.layer, seg.width_mm, neck), [0.0, words])
                g[0] = float(g[0]) + math.hypot(t.end[0] - t.start[0], t.end[1] - t.start[1])
                if not g[1]:
                    g[1] = words
        for (layer, w, neck), (length, words) in groups.items():
            row: dict[str, Any] = {"net": net, "layer": layer, "width_mm": w, "length_mm": round(float(length), 6)}
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
    details: dict[str, Any] = {"min_width_mm": need, "source": cls.min_width_mm.provenance.tool or cls.min_width_mm.provenance.kind.value, "nets": nets}
    if not tracks:
        return _result(check, S.NOT_APPLICABLE, f"class {cls.name}: no routed track", **details)
    narrow = sorted({(t.net, t.layer, float(t.width_mm)) for t in tracks if float(t.width_mm) < need - _TOL})
    details["narrow"] = [{"net": n, "layer": layer, "width_mm": w} for n, layer, w in narrow]
    if narrow:
        details["repair"] = "human"
        return _result(check, S.FAIL, f"class {cls.name}: track(s) narrower than the minimum {need:.6g} mm: " + ", ".join(f"{n} {layer} {w:g} mm" for n, layer, w in narrow), **details)
    return _result(check, S.PASS, f"class {cls.name}: {len(tracks)} track(s) of {len(set(t.net for t in tracks))} net(s) at least {need:.6g} mm wide", **details)


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
            if not pieces:
                rows.append({"status": S.FAIL.value, "reason": "no coupled section: the two nets never run side by side"})
            if cls.pair_uncoupled_max_mm is not None:
                budget = float(cls.pair_uncoupled_max_mm.value)
                for net, unc in ((pr.p, unc_p), (pr.n, unc_n)):
                    ok = unc <= budget + 1e-6
                    rows.append({"net": net, "uncoupled_mm": round(unc, 6), "budget_mm": budget, "status": (S.PASS if ok else S.FAIL).value,
                                 "reason": f"{net} uncoupled {unc:.3f} mm {'<=' if ok else '>'} {budget:g} mm"})
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
    out.extend(timing_results(ir, si, measures))
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
    "critical_result",
    "delay_result",
    "diff_results",
    "impedance_result",
    "length_result",
    "rf_length_result",
    "si_results",
    "skew_results",
    "timing_results",
    "width_result",
]
