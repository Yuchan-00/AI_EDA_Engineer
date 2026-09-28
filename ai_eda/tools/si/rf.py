"""Which nets carry RF, at which frequency, and the RF electrical-length rule on their routed copper.

Invariant: a net is an RF net only because the IR says so - a net of kind
``NetKind.RF``, or a member of a net class that states ``rf_frequency_hz`` -
and its frequency only comes from the IR: the class's ``rf_frequency_hz``,
else the design's confirmed ``carrier_frequency`` requirement (read through
:func:`ai_eda.design.inputs.read_value`, so an unconfirmed or ambiguous value
is no frequency). Nothing is guessed from a net's name, and nothing here is a
verdict: :mod:`ai_eda.validation.si` (``si.rf_length``) and
:mod:`ai_eda.validation.domain` (``domain.rf.impedance``) judge.

The RF electrical-length rule (:func:`rf_length_rows`) is the carrier form of
the critical-length rule of :mod:`ai_eda.tools.si.promote`: a net's *line* is
electrically short when its length is at most

    l_crit = fraction * lambda_g,   lambda_g = 1 / (f t_pd)      (``calc.tline``: :func:`~ai_eda.tools.calc.tline.rf_critical_length_mm`)

with ``fraction`` ``ir.si.rf_length_fraction`` (a confirmed choice; 0.1 is
the common lambda/10 rule of thumb - never defaulted). The line, its delay
per length and the bound flag are the ones :class:`~ai_eda.tools.si.measure.NetMeasure`
gives the rise-time rule (``m.line``: the longest pad-to-pad path or a 2-pad
net's copper; else the whole copper, an upper bound), so ``si.rf_length``
and ``si.critical_length`` read one t_pd per net. A bound is used only in the
direction it holds: without a reference plane t_pd is the upper bound
sqrt(er)/c0, so l_crit is a lower bound - a line shorter than it is short, a
longer one only "possibly long"; the whole copper of a multi-pad net is an
upper bound of its line - shorter than l_crit is short, longer is "possibly
long". Nothing is promoted or re-routed by this rule (the router is
unchanged): it informs.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_eda.ir import CircuitDomain, CircuitIR, NetKind, Traced
from ai_eda.tools.calc.tline import guided_wavelength_mm, rf_critical_length_mm
from ai_eda.tools.si.measure import NetMeasure

#: the check ids this module's rows and nets feed
RF_LENGTH_CHECK = "si.rf_length"
RF_IMPEDANCE_CHECK = "domain.rf.impedance"
#: the canonical requirement key of the design's carrier (``ai_eda.design.inputs``)
CARRIER_KEY = "carrier_frequency"


def rf_nets(ir: CircuitIR) -> list[str]:
    """The nets that carry RF, in IR order: kind ``rf``, or routed by a net class that states ``rf_frequency_hz``."""
    si = ir.si
    out: list[str] = []
    for n in ir.nets:
        cls = si.class_of(n.name) if si is not None else None
        if n.kind == NetKind.RF or (cls is not None and cls.rf_frequency_hz is not None):
            out.append(n.name)
    return out


def rf_domain(ir: CircuitIR) -> bool:
    """Whether the topology declares the RF domain."""
    return ir.topology is not None and CircuitDomain.RF in ir.topology.domains


def has_rf(ir: CircuitIR) -> bool:
    """Whether the design carries RF: an RF net (:func:`rf_nets`) or the RF domain in its topology."""
    return bool(rf_nets(ir)) or rf_domain(ir)


def format_hz(f: float) -> str:
    """``900 MHz``, ``2.4 GHz``, ``455 kHz`` - for messages."""
    for scale, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if abs(f) >= scale:
            return f"{f / scale:.6g} {unit}"
    return f"{f:.6g} Hz"


def design_carrier_frequency(ir: CircuitIR) -> tuple[Traced | None, str | None]:
    """``(traced Hz, None)`` of the design's confirmed ``carrier_frequency`` requirement, or ``(None, why)``.

    Every confirmed requirement under the key's aliases must read (through
    :func:`ai_eda.design.inputs.read_value`) to the same positive number;
    none, an unreadable one or two different numbers ("ambiguous") is no
    frequency.
    """
    # the ai_eda.design package imports ai_eda.tools.si (stackup / rules): a module-level import would be circular
    from ai_eda.design.inputs import KEY_ALIASES, read_value

    aliases = KEY_ALIASES.get(CARRIER_KEY, (CARRIER_KEY,))
    candidates = [r for r in ir.requirements.requirements if r.key in aliases]
    if not candidates:
        return None, f"no {CARRIER_KEY} requirement"
    readings: list[tuple[str, Traced]] = []
    reasons: list[str] = []
    for r in candidates:
        t, why = read_value(r, "Hz")
        if t is None:
            reasons.append(why or f"{r.id}: unreadable")
        elif not float(t.value) > 0.0:
            reasons.append(f"{r.id}: a carrier frequency must be > 0 Hz, got {float(t.value):.6g} Hz")
        else:
            readings.append((r.id, t))
    if reasons:
        return None, f"{CARRIER_KEY} is not usable: " + "; ".join(reasons)
    first = readings[0][1]
    if any(float(t.value) != float(first.value) for _, t in readings[1:]):
        return None, f"{CARRIER_KEY} is ambiguous: " + ", ".join(f"{rid} says {float(t.value):.12g} Hz" for rid, t in readings)
    return first, None


def rf_frequency_for(ir: CircuitIR, net: str) -> tuple[float | None, str]:
    """``(f Hz, where it came from)`` for an RF net: its class's ``rf_frequency_hz``, else the design's carrier; else ``(None, why)``."""
    cls = ir.si.class_of(net) if ir.si is not None else None
    if cls is not None and cls.rf_frequency_hz is not None:
        return float(cls.rf_frequency_hz.value), f"si.net_classes[{cls.name}].rf_frequency_hz ({cls.rf_frequency_hz.provenance.kind.value})"
    carrier, why = design_carrier_frequency(ir)
    if carrier is not None:
        return float(carrier.value), f"the confirmed {CARRIER_KEY} ({carrier.provenance.derived_from[0]})"
    owner = f"class {cls.name} states" if cls is not None else f"{net} is in no net class that states"
    return None, f"no frequency: neither {owner} rf_frequency_hz nor is a {CARRIER_KEY} requirement confirmed ({why})"


@dataclass
class RfLengthRow:
    """One RF net under the RF electrical-length rule: ``status`` short / long / possibly_long / unknown / not_applicable, the reason and the numbers.

    ``long`` is a line over a reference plane (its t_pd computed, not bounded)
    longer than l_crit; ``possibly_long`` is a length over l_crit that rests on
    a bound (the no-plane t_pd, or the whole copper of a multi-pad net).
    """

    net: str
    net_class: str | None
    status: str
    reason: str = ""
    f_hz: float | None = None
    f_source: str = ""
    fraction: float | None = None
    length_mm: float | None = None
    measure: str = ""
    t_pd_s_per_m: float | None = None
    lambda_g_mm: float | None = None
    l_crit_mm: float | None = None
    bound: bool = False
    whole_copper: bool = False

    def as_dict(self) -> dict:
        return {
            "net": self.net, "class": self.net_class, "status": self.status, "reason": self.reason,
            "f_hz": self.f_hz, "f_source": self.f_source, "fraction": self.fraction,
            "length_mm": None if self.length_mm is None else round(self.length_mm, 6), "measure": self.measure,
            "t_pd_ps_per_mm": None if self.t_pd_s_per_m is None else round(self.t_pd_s_per_m * 1e9, 6),
            "lambda_g_mm": None if self.lambda_g_mm is None else round(self.lambda_g_mm, 6),
            "l_crit_mm": None if self.l_crit_mm is None else round(self.l_crit_mm, 6),
            "bound": self.bound, "whole_copper": self.whole_copper,
        }


def rf_length_rows(ir: CircuitIR, measures: dict[str, NetMeasure]) -> list[RfLengthRow]:
    """One :class:`RfLengthRow` per RF net (:func:`rf_nets`), in IR order (module docstring)."""
    si = ir.si
    fraction = None if si is None or si.rf_length_fraction is None else float(si.rf_length_fraction.value)
    out: list[RfLengthRow] = []
    for net in rf_nets(ir):
        cls = si.class_of(net) if si is not None else None
        row = RfLengthRow(net=net, net_class=None if cls is None else cls.name, status="unknown", fraction=fraction)
        out.append(row)
        f, source = rf_frequency_for(ir, net)
        if f is None:
            row.reason = source
            continue
        row.f_hz, row.f_source = f, source
        m = measures.get(net)
        if m is None or not m.routed:
            if m is not None and m.pads < 2:
                row.status, row.reason = "not_applicable", "fewer than two pads: nothing routed"
            else:
                row.reason = "no routed copper"
            continue
        if fraction is None:
            row.reason = "rf_length_fraction is not stated (ir.si.rf_length_fraction: a confirmed choice, e.g. 0.1 for the lambda/10 rule)"
            continue
        line = m.line
        if line is not None:
            row.length_mm, row.measure, row.t_pd_s_per_m, row.bound = line.length_mm, line.describe(), line.t_pd_s_per_m, line.bound
        else:
            row.length_mm, row.t_pd_s_per_m, row.bound, row.whole_copper = m.length_mm, m.t_pd_s_per_m, m.bound, True
            row.measure = "the whole copper (an upper bound of every pad-to-pad path)"
        if row.t_pd_s_per_m is None:
            row.reason = "; ".join(m.problems) or "no delay per length"
            continue
        row.lambda_g_mm = guided_wavelength_mm(f, row.t_pd_s_per_m)
        row.l_crit_mm = rf_critical_length_mm(f, row.t_pd_s_per_m, fraction)
        at = f"at {format_hz(f)} ({fraction:g} lambda_g rule, lambda_g {row.lambda_g_mm:.3f} mm, not a simulation)"
        if row.length_mm <= row.l_crit_mm:
            row.status = "short"
            row.reason = f"electrically short {at}: {row.measure} {row.length_mm:.3f} mm <= l_crit {row.l_crit_mm:.3f} mm" + (
                " (with the no-plane upper bound of t_pd)" if row.bound else "")
        elif row.whole_copper:
            row.status = "possibly_long"
            row.reason = (f"possibly long {at}: the whole copper of its {m.pads} pads ({row.length_mm:.3f} mm, an upper bound of every path) exceeds "
                          f"l_crit {row.l_crit_mm:.3f} mm, but no pad-to-pad path was extracted ({m.path_problem or 'not tried'})")
            if row.bound:
                row.reason += "; impedance is undefined without a reference plane - use pcb_layers=4 or add a plane"
        elif row.bound:
            row.status = "possibly_long"
            row.reason = (f"possibly long {at}: {row.measure} {row.length_mm:.3f} mm > l_crit {row.l_crit_mm:.3f} mm by the no-plane upper bound of t_pd; "
                          "impedance is undefined without a reference plane - use pcb_layers=4 or add a plane")
        else:
            row.status = "long"
            row.reason = f"electrically long {at}: {row.measure} {row.length_mm:.3f} mm > l_crit {row.l_crit_mm:.3f} mm over the reference plane"
    return out


__all__ = [
    "CARRIER_KEY",
    "RF_IMPEDANCE_CHECK",
    "RF_LENGTH_CHECK",
    "RfLengthRow",
    "design_carrier_frequency",
    "format_hz",
    "has_rf",
    "rf_domain",
    "rf_frequency_for",
    "rf_length_rows",
    "rf_nets",
]
