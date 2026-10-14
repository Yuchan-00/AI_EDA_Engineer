"""The critical-length rule: which routed nets are electrically long, and their promotion to a controlled-impedance class.

Invariant: a net is promoted only by the stated rule on measured numbers -
never by a guess about what "looks fast". After a routing pass, a net is
*electrically long* when the delay of its *line* exceeds ``fraction * t_r``,
equivalently when the line's length exceeds

    l_crit = fraction * t_r / t_pd     (``calc.tline.critical_length``)

with ``t_r`` the edge of its class's driver (:mod:`ai_eda.tools.si.driver`:
a grounded datasheet ``t_rise`` of a driver that drives the net, or the
class's confirmed choice), ``fraction`` ``ir.si.critical_fraction`` (1/2:
the round trip is shorter than the edge) and ``t_pd`` the line's own delay
per length from :mod:`ai_eda.tools.si.measure` (over a plane the
microstrip's; without one the upper bound sqrt(er)/c0, which can only make a
net look longer). The line (:attr:`~ai_eda.tools.si.measure.NetMeasure.line`)
is a line that exists in the copper: the longest pad-to-pad path
(:mod:`ai_eda.tools.si.paths`), or the whole copper of a 2-pad net. The
whole copper of a net with more pads is only an upper bound of every path
through it - summed branches are not a line - so it is used only in the
direction a bound is valid: when even the total is shorter than l_crit the
net is short; otherwise, with no path extracted, the net is ``possibly_long``
(not judged, never promoted). The rule does not apply to a supply or ground
net (``NetKind.POWER`` / ``GROUND``: no driven edge), to a net whose class
has no driver edge, or to a net already in a controlled-impedance class.

:func:`promote` moves every electrically long net whose class names
``promote_to`` into that class: one :class:`~ai_eda.ir.Promotion` with
``derived`` provenance (tool :data:`PROMOTE_TOOL`) naming the rule, the
measured length and delay, l_crit and where t_r and the stackup numbers came
from. It returns a new :class:`~ai_eda.ir.SIConstraints`; the IR is only read.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_eda.ir import CircuitIR, NetKind, Promotion, Provenance, ProvenanceKind, SIConstraints
from ai_eda.tools.calc.tline import critical_length_mm
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.si.driver import driver_value
from ai_eda.tools.si.measure import SI_VERSION, NetMeasure, measure_nets

#: provenance ``tool`` of a promotion
PROMOTE_TOOL = "si.promote"
#: net kinds the critical-length rule never applies to (no driven edge)
UNDRIVEN_KINDS = frozenset({NetKind.POWER, NetKind.GROUND})


@dataclass
class CriticalRow:
    """One net under the critical-length rule: the numbers, and ``status`` short / long / possibly_long / not_applicable / unknown with ``reason``.

    ``length_mm`` / ``delay_s`` are the line's (``measure`` says which line:
    the longest pad-to-pad path ``ends``, a 2-pad net's copper, or - for a
    short or possibly long net without a line - the whole copper, an upper
    bound); ``total_mm`` is always the whole copper.
    """

    net: str
    net_class: str | None
    status: str
    reason: str = ""
    length_mm: float | None = None
    delay_s: float | None = None
    t_pd_s_per_m: float | None = None
    l_crit_mm: float | None = None
    t_rise_s: float | None = None
    t_rise_source: str = ""
    fraction: float | None = None
    bound: bool = False
    measure: str = ""
    ends: tuple[str, str] | None = None
    total_mm: float | None = None

    def as_dict(self) -> dict:
        return {
            "net": self.net, "class": self.net_class, "status": self.status, "reason": self.reason,
            "length_mm": None if self.length_mm is None else round(self.length_mm, 6),
            "measure": self.measure, "ends": None if self.ends is None else list(self.ends),
            "total_mm": None if self.total_mm is None else round(self.total_mm, 6),
            "delay_ps": None if self.delay_s is None else round(self.delay_s * 1e12, 4),
            "t_pd_ps_per_mm": None if self.t_pd_s_per_m is None else round(self.t_pd_s_per_m * 1e9, 6),
            "l_crit_mm": None if self.l_crit_mm is None else round(self.l_crit_mm, 6),
            "t_rise_s": self.t_rise_s, "t_rise_source": self.t_rise_source, "fraction": self.fraction, "bound": self.bound,
        }


def critical_rows(ir: CircuitIR, measures: dict[str, NetMeasure] | None = None, library: KicadLibrary | None = None) -> list[CriticalRow]:
    """One :class:`CriticalRow` per IR net, in IR order (module docstring); ``library`` extracts the paths when ``measures`` is not given."""
    si = ir.si
    measures = measure_nets(ir, library=library) if measures is None else measures
    out: list[CriticalRow] = []
    for net in ir.nets:
        cls = si.class_of(net.name) if si is not None else None
        row = CriticalRow(net=net.name, net_class=None if cls is None else cls.name, status="not_applicable")
        out.append(row)
        if net.kind in UNDRIVEN_KINDS:
            row.reason = f"a {net.kind.value} net carries no driven edge"
            continue
        if si is None or cls is None:
            row.reason = "no net class (ir.si) states a driver edge"
            continue
        t_r, missing = driver_value(ir, cls, "t_rise_s", net.name)
        if t_r is None:
            row.reason = f"class {cls.name} states no driver edge ({missing})"
            continue
        row.t_rise_s, row.t_rise_source = t_r.value, t_r.source
        m = measures.get(net.name)
        if m is None or not m.routed:
            row.status, row.reason = "unknown", "no routed copper"
            continue
        row.total_mm = row.length_mm = m.length_mm
        if si.critical_fraction is None:
            row.status, row.reason = "unknown", "ir.si.critical_fraction is not stated"
            continue
        row.fraction = float(si.critical_fraction.value)
        line = m.line
        if line is None:  # more than two pads and no path: only the whole copper's upper bound is known
            t_pd = m.t_pd_s_per_m
            if t_pd is None:
                row.status, row.reason = "unknown", "; ".join(m.problems) or "no delay"
                continue
            row.delay_s, row.t_pd_s_per_m, row.bound = m.delay_s, t_pd, m.bound
            row.l_crit_mm = critical_length_mm(t_r.value, t_pd, row.fraction)
            row.measure = "the whole copper (an upper bound of every pad-to-pad path)"
            if row.length_mm <= row.l_crit_mm:
                row.status = "short"
            else:
                row.status = "possibly_long"
                row.reason = (f"the whole copper of its {m.pads} pads ({row.length_mm:.3f} mm, an upper bound of every path) exceeds l_crit {row.l_crit_mm:.3f} mm, "
                              f"but no pad-to-pad path was extracted ({m.path_problem or 'not tried'}): not judged, not promoted")
            continue
        t_pd = line.t_pd_s_per_m
        row.length_mm, row.measure, row.ends = line.length_mm, line.describe(), line.ends
        if t_pd is None:
            row.status, row.reason = "unknown", "; ".join(m.problems) or "no delay"
            continue
        row.delay_s, row.t_pd_s_per_m, row.bound = line.delay_s, t_pd, line.bound
        row.l_crit_mm = critical_length_mm(t_r.value, t_pd, row.fraction)
        row.status = "long" if row.length_mm > row.l_crit_mm else "short"
    return out


def promote(
    ir: CircuitIR, measures: dict[str, NetMeasure] | None = None, library: KicadLibrary | None = None,
) -> tuple[SIConstraints | None, list[Promotion], list[str]]:
    """``(new ir.si, promotions, notes)``: every electrically long net whose class names ``promote_to`` moved there (module docstring).

    ``new ir.si`` is ``None`` when nothing was promoted (the IR's own stays).
    ``library`` (the KiCad footprints on disk) lets the rule extract the
    pad-to-pad paths; without it a net with more than two pads can only be
    found short (by its whole copper) or possibly long, never promoted.
    """
    si = ir.si
    if si is None:
        return None, [], []
    rows = critical_rows(ir, measures, library)
    added: dict[str, list[Promotion]] = {}
    notes: list[str] = []
    stackup = ir.pcb.stackup if ir.pcb is not None else None
    stack_ids = [k for k, _ in stackup.traced_items()] if stackup is not None else []
    for row in rows:
        if row.status == "possibly_long":
            notes.append(f"{row.net} not promoted: {row.reason}")
            continue
        if row.status != "long":
            continue
        cls = si.class_of(row.net)
        assert cls is not None
        if cls.target_z0_ohm is not None:
            continue  # already controlled
        if cls.promote_to is None:
            notes.append(f"{row.net} is electrically long ({row.length_mm:.3f} mm > l_crit {row.l_crit_mm:.3f} mm) but class {cls.name} names no controlled class to promote to")
            continue
        bound = " (t_pd is the no-plane upper bound sqrt(er)/c0)" if row.bound else ""
        note = (
            f"critical-length rule: {row.measure} {row.length_mm:.4f} mm (delay {row.delay_s * 1e12:.2f} ps at t_pd {row.t_pd_s_per_m * 1e9:.4f} ps/mm{bound}; "
            f"the whole copper {row.total_mm:.4f} mm) "
            f"> l_crit = {row.fraction:g} x t_r {row.t_rise_s:g} s / t_pd = {row.l_crit_mm:.4f} mm (t_r from {row.t_rise_source}); "
            f"promoted from {cls.name} to {cls.promote_to}"
        )
        prov = Provenance(
            kind=ProvenanceKind.DERIVED, tool=PROMOTE_TOOL, tool_version=SI_VERSION,
            derived_from=[f"pcb.tracks[net={row.net}]", f"pcb.vias[net={row.net}]", row.t_rise_source.split(" ")[0], "si.critical_fraction", *stack_ids],
            note=note,
        )
        added.setdefault(cls.promote_to, []).append(Promotion(
            net=row.net, from_class=cls.name, length_mm=round(row.length_mm, 6), delay_s=float(f"{row.delay_s:.9e}"),
            l_crit_mm=round(row.l_crit_mm, 6), t_rise_s=row.t_rise_s, provenance=prov,
        ))
        notes.append(f"{row.net}: {note}")
    if not added:
        return None, [], notes
    classes = [c.model_copy(update={"promoted": [*c.promoted, *added.get(c.name, [])]}) if c.name in added else c for c in si.net_classes]
    new = SIConstraints.model_validate({**si.model_dump(), "net_classes": [c.model_dump() for c in classes]})
    return new, [p for ps in added.values() for p in ps], notes


__all__ = ["PROMOTE_TOOL", "UNDRIVEN_KINDS", "CriticalRow", "critical_rows", "promote"]
