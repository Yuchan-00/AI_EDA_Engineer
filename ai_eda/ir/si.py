"""Signal-integrity constraints of the design (``ir.si``): net classes and timing paths.

Invariant: this is *design content* - what the design asks of its copper -
and every number in it is :class:`~ai_eda.ir.provenance.Traced`: a template's
confirmed choice, the user's value, a grounded datasheet / fab fact or a
registered calculator's output (``derived``, re-derived by ``calc.recompute``
through the ids :meth:`SIConstraints.traced_items` names). Nothing here is a
verdict: the ``si.*`` checks (:mod:`ai_eda.validation.si`) and ``spice.si``
judge the routed copper against it, and a number they cannot get (no
stackup, no reference plane, a datasheet timing key nobody grounded) makes
them NOT_VERIFIED naming it. ``CircuitIR.si`` is ``None`` for a design that
states no SI constraint and is then left out of the design view, so an IR
saved before the field existed keeps its hash.

* :class:`NetClass` - a named set of nets and what their copper must meet:
  a single-ended impedance target (``target_z0_ohm`` +/- ``z0_tol_rel``), a
  differential target for its declared ``pairs`` (``target_zdiff_ohm`` +/-
  ``zdiff_tol_rel``, the uncoupled-length budget ``pair_uncoupled_max_mm``
  and the intra-pair skew ``pair_max_skew_s``), a routed length / delay
  budget, a match group with its skew budget, a minimum track width
  (``min_width_mm``: a current-capacity width such as IPC-2221's) and the
  driver model the critical-length rule and the SPICE check use (``t_rise_s``,
  ``r_drive_ohm``, ``c_load_f``, ``ringing_tol_rel``; a ``driver`` component
  whose grounded datasheet facts ``t_rise`` / ``r_out`` / ``c_in`` replace
  them on the nets the driver drives - an output / bidirectional pin there).
  One class may be the ``default``: the class of every net no class lists.
  ``promote_to`` names the controlled-impedance class a net of this class
  moves to when its line's routed delay exceeds the critical length; each such
  move is a :class:`Promotion` in the target class's ``promoted`` list, with
  ``derived`` provenance naming the rule, the measured delay and l_crit.
* :class:`TimingPath` - a synchronous interface: a clock net, the data nets it
  captures, the clock frequency, the fraction of the period between the
  launching and the capturing edge, and the four datasheet terms
  (``t_co_max_s`` / ``t_co_min_s`` / ``t_su_min_s`` / ``t_h_min_s``). A term
  may be given directly or named in ``terms_from`` as ``<ref>.<fact key>``
  (``U1.t_su``): it is then read from that component's grounded datasheet
  facts at check time, and missing until someone grounds it.
* ``critical_fraction`` - the fraction of the rise time in the
  critical-length rule l_crit = fraction * t_r / t_pd (a confirmed choice;
  1/2 is the round-trip rule).
"""

from __future__ import annotations

import math
import re

from pydantic import BaseModel, Field, model_validator

from ai_eda.ir.provenance import Provenance, Traced

#: the id prefix of every traced SI number (``si.net_classes[Z50].target_z0_ohm``), resolved by :meth:`SIConstraints.lookup`
SI_PREFIX = "si"
#: ``<ref>.<fact key>`` in ``TimingPath.terms_from`` / ``NetClass.driver`` facts
FACT_REF_RE = re.compile(r"^(?P<ref>[A-Za-z][A-Za-z0-9_]*)\.(?P<key>[a-z][a-z0-9_]*)$")
#: timing terms a path may name, with their unit
TIMING_TERMS: dict[str, str] = {"t_co_max_s": "s", "t_co_min_s": "s", "t_su_min_s": "s", "t_h_min_s": "s"}
#: the unit each traced NetClass field must carry (``None``: a dimensionless ratio)
CLASS_UNITS: dict[str, str | None] = {
    "target_z0_ohm": "ohm", "z0_tol_rel": None, "target_zdiff_ohm": "ohm", "zdiff_tol_rel": None, "pair_uncoupled_max_mm": "mm",
    "pair_max_skew_s": "s", "max_length_mm": "mm", "max_delay_s": "s", "max_skew_s": "s", "min_width_mm": "mm", "power_current_a": "A",
    "power_temp_rise_c": "degC", "t_rise_s": "s", "r_drive_ohm": "ohm", "c_load_f": "F", "ringing_tol_rel": None,
}
#: fields of a class that must be > 0 when given (the rest: >= 0)
_POSITIVE = frozenset({"target_z0_ohm", "target_zdiff_ohm", "max_length_mm", "max_delay_s", "min_width_mm", "power_current_a", "power_temp_rise_c",
                       "t_rise_s", "c_load_f", "z0_tol_rel", "zdiff_tol_rel", "ringing_tol_rel"})
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_+\-]*$")


def _check_number(t: Traced | None, what: str, unit: str | None, positive: bool) -> None:
    if t is None:
        return
    if t.unit != unit:
        want = f"unit {unit!r}" if unit is not None else "no unit (a dimensionless ratio)"
        raise ValueError(f"{what} must carry {want}, got {t.unit!r}")
    v = t.value
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{what} must be a finite number, got {v!r}")
    if (positive and not v > 0) or (not positive and v < 0):
        raise ValueError(f"{what} must be {'> 0' if positive else '>= 0'}, got {v!r}")


class DiffPair(BaseModel):
    """A declared differential pair: the positive and the negative net (both in the class's ``nets``)."""

    p: str
    n: str

    @property
    def name(self) -> str:
        return f"{self.p}/{self.n}"


class Promotion(BaseModel):
    """A net moved into a controlled-impedance class by the critical-length rule after the first routing pass.

    The numbers are the measurement that triggered it (the length of the
    net's line - its longest pad-to-pad path through the copper, via barrels
    included, or a 2-pad net's copper - the delay from the stackup's
    propagation delay per length, the critical length of the net's class);
    ``provenance`` is
    ``derived`` by the promotion rule (``tool`` ``si.promote``) and its note
    states the rule with the numbers.
    """

    net: str
    from_class: str
    length_mm: float
    delay_s: float
    l_crit_mm: float
    t_rise_s: float
    provenance: Provenance

    @model_validator(mode="after")
    def _finite(self) -> Promotion:
        for name in ("length_mm", "delay_s", "l_crit_mm", "t_rise_s"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"promotion of {self.net}: {name} must be a finite number >= 0, got {v!r}")
        return self


class NetClass(BaseModel):
    """A named set of nets and the SI constraints their copper must meet (module docstring)."""

    name: str
    description: str = ""
    nets: list[str] = Field(default_factory=list)
    #: the class of every net no class lists (at most one class is the default)
    default: bool = False
    target_z0_ohm: Traced[float] | None = None
    z0_tol_rel: Traced[float] | None = None
    target_zdiff_ohm: Traced[float] | None = None
    zdiff_tol_rel: Traced[float] | None = None
    pairs: list[DiffPair] = Field(default_factory=list)
    pair_uncoupled_max_mm: Traced[float] | None = None
    pair_max_skew_s: Traced[float] | None = None
    max_length_mm: Traced[float] | None = None
    max_delay_s: Traced[float] | None = None
    match_group: str | None = None
    max_skew_s: Traced[float] | None = None
    #: a minimum track width (``calc.ipc2221.width_for_current`` of ``power_current_a`` at ``power_temp_rise_c``)
    min_width_mm: Traced[float] | None = None
    power_current_a: Traced[float] | None = None
    power_temp_rise_c: Traced[float] | None = None
    #: the driver model: the edge for the critical-length rule and the SPICE check, the source resistance, the far-end load
    t_rise_s: Traced[float] | None = None
    r_drive_ohm: Traced[float] | None = None
    c_load_f: Traced[float] | None = None
    #: overshoot / undershoot / settling band of the SPICE check, as a fraction of the swing
    ringing_tol_rel: Traced[float] | None = None
    #: the component whose grounded datasheet facts ``t_rise`` / ``r_out`` / ``c_in`` replace the driver choices above on the nets it drives (an output / bidirectional pin there)
    driver: str | None = None
    #: the controlled-impedance class a net of this class is promoted to when it is electrically long
    promote_to: str | None = None
    promoted: list[Promotion] = Field(default_factory=list)
    provenance: Provenance

    @model_validator(mode="after")
    def _consistent(self) -> NetClass:
        if not _NAME_RE.match(self.name):
            raise ValueError(f"net class name {self.name!r} must start with a letter and use letters, digits, _ + - only")
        for field, unit in CLASS_UNITS.items():
            _check_number(getattr(self, field), f"net class {self.name}: {field}", unit, field in _POSITIVE)
        if len(set(self.nets)) != len(self.nets):
            raise ValueError(f"net class {self.name}: a net is listed twice in {self.nets}")
        if (self.match_group is None) != (self.max_skew_s is None):
            raise ValueError(f"net class {self.name}: match_group and max_skew_s go together")
        if self.target_z0_ohm is not None and self.z0_tol_rel is None:
            raise ValueError(f"net class {self.name}: a target_z0_ohm needs z0_tol_rel")
        if self.target_zdiff_ohm is not None and self.zdiff_tol_rel is None:
            raise ValueError(f"net class {self.name}: a target_zdiff_ohm needs zdiff_tol_rel")
        if self.pairs and self.target_zdiff_ohm is None:
            raise ValueError(f"net class {self.name}: declared pairs need a target_zdiff_ohm")
        paired: list[str] = []
        for pr in self.pairs:
            if pr.p == pr.n:
                raise ValueError(f"net class {self.name}: pair {pr.name} names one net twice")
            for x in (pr.p, pr.n):
                if x not in self.nets:
                    raise ValueError(f"net class {self.name}: pair net {x!r} is not in the class's nets")
                paired.append(x)
        if len(set(paired)) != len(paired):
            raise ValueError(f"net class {self.name}: a net is in two pairs")
        if self.min_width_mm is not None and self.min_width_mm.provenance.tool == "calc.ipc2221.width_for_current":
            if self.power_current_a is None or self.power_temp_rise_c is None:
                raise ValueError(f"net class {self.name}: an IPC-2221 min_width_mm needs power_current_a and power_temp_rise_c")
        if self.promote_to == self.name:
            raise ValueError(f"net class {self.name}: promote_to names the class itself")
        if self.promoted and self.target_z0_ohm is None:
            raise ValueError(f"net class {self.name}: only a controlled-impedance class (target_z0_ohm) takes promoted nets")
        seen = [p.net for p in self.promoted]
        if len(set(seen)) != len(seen) or set(seen) & set(self.nets):
            raise ValueError(f"net class {self.name}: a promoted net is listed twice or is already a declared member")
        return self

    def members(self) -> list[str]:
        """Declared nets, then promoted ones."""
        return [*self.nets, *(p.net for p in self.promoted)]


class TimingPath(BaseModel):
    """A synchronous interface: a clock net launching / capturing data nets (module docstring)."""

    name: str
    clock_net: str
    data_nets: list[str]
    #: who launches and who captures, in words (``"ISP programmer at J2 -> U1"``)
    direction: str = ""
    f_clk_hz: Traced[float] | None = None
    #: the time from the launching to the capturing clock edge as a fraction of the period (1: the same edge one period
    #: later; 0.5: opposite edges, as SPI mode 0 launches on the falling and captures on the rising edge)
    capture_fraction: Traced[float] | None = None
    t_co_max_s: Traced[float] | None = None
    t_co_min_s: Traced[float] | None = None
    t_su_min_s: Traced[float] | None = None
    t_h_min_s: Traced[float] | None = None
    #: term -> ``<ref>.<fact key>``: where a datasheet fact would ground a term that is ``None`` here
    terms_from: dict[str, str] = Field(default_factory=dict)
    provenance: Provenance

    @model_validator(mode="after")
    def _consistent(self) -> TimingPath:
        if not _NAME_RE.match(self.name):
            raise ValueError(f"timing path name {self.name!r} must start with a letter and use letters, digits, _ + - only")
        if not self.data_nets or self.clock_net in self.data_nets:
            raise ValueError(f"timing path {self.name}: needs at least one data net other than the clock {self.clock_net!r}")
        _check_number(self.f_clk_hz, f"timing path {self.name}: f_clk_hz", "Hz", True)
        _check_number(self.capture_fraction, f"timing path {self.name}: capture_fraction", None, True)
        if self.capture_fraction is not None and self.capture_fraction.value > 1.0:
            raise ValueError(f"timing path {self.name}: capture_fraction must be in (0, 1], got {self.capture_fraction.value!r}")
        for term, unit in TIMING_TERMS.items():
            _check_number(getattr(self, term), f"timing path {self.name}: {term}", unit, False)
        for term, ref in self.terms_from.items():
            if term not in TIMING_TERMS:
                raise ValueError(f"timing path {self.name}: terms_from names {term!r}, which is not one of {sorted(TIMING_TERMS)}")
            if not FACT_REF_RE.match(ref):
                raise ValueError(f"timing path {self.name}: terms_from[{term}] = {ref!r} is not '<ref>.<fact key>'")
        return self


class SIConstraints(BaseModel):
    """``ir.si``: the net classes, the timing paths and the critical-length rule's fraction (module docstring)."""

    net_classes: list[NetClass] = Field(default_factory=list)
    timing_paths: list[TimingPath] = Field(default_factory=list)
    critical_fraction: Traced[float] | None = None
    provenance: Provenance

    @model_validator(mode="after")
    def _consistent(self) -> SIConstraints:
        names = [c.name for c in self.net_classes]
        if len(set(names)) != len(names):
            raise ValueError(f"net class names must be unique, got {names}")
        defaults = [c.name for c in self.net_classes if c.default]
        if len(defaults) > 1:
            raise ValueError(f"at most one default net class, got {defaults}")
        owner: dict[str, str] = {}
        for c in self.net_classes:
            for net in c.nets:
                if net in owner:
                    raise ValueError(f"net {net!r} is in two classes ({owner[net]}, {c.name})")
                owner[net] = c.name
        moved: dict[str, str] = {}
        for c in self.net_classes:
            for p in c.promoted:
                if p.net in moved:
                    raise ValueError(f"net {p.net!r} is promoted twice ({moved[p.net]}, {c.name})")
                moved[p.net] = c.name
                if p.net in owner and owner[p.net] != p.from_class:
                    raise ValueError(f"promotion of {p.net} into {c.name}: the net is declared in {owner[p.net]}, not in {p.from_class}")
        by_name = {c.name: c for c in self.net_classes}
        for c in self.net_classes:
            if c.promote_to is not None:
                target = by_name.get(c.promote_to)
                if target is None or target.target_z0_ohm is None:
                    raise ValueError(f"net class {c.name}: promote_to {c.promote_to!r} is not a controlled-impedance class of this design")
        for c in self.net_classes:
            for p in c.promoted:
                if p.from_class not in by_name or by_name[p.from_class].promote_to != c.name:
                    raise ValueError(f"promotion of {p.net} into {c.name}: class {p.from_class!r} does not promote to {c.name}")
        groups: dict[str, float] = {}
        for c in self.net_classes:
            if c.match_group is not None:
                skew = float(c.max_skew_s.value)  # type: ignore[union-attr]
                if groups.setdefault(c.match_group, skew) != skew:
                    raise ValueError(f"match group {c.match_group}: its classes disagree on max_skew_s")
        paths = [p.name for p in self.timing_paths]
        if len(set(paths)) != len(paths):
            raise ValueError(f"timing path names must be unique, got {paths}")
        _check_number(self.critical_fraction, "si.critical_fraction", None, True)
        if self.critical_fraction is not None and self.critical_fraction.value > 1.0:
            raise ValueError(f"si.critical_fraction must be in (0, 1], got {self.critical_fraction.value!r}")
        return self

    # --- lookups -------------------------------------------------------------

    def net_class(self, name: str) -> NetClass | None:
        for c in self.net_classes:
            if c.name == name:
                return c
        return None

    def default_class(self) -> NetClass | None:
        return next((c for c in self.net_classes if c.default), None)

    def class_of(self, net: str) -> NetClass | None:
        """The class that routes ``net``: the class it was promoted to, else the class listing it, else the default class."""
        for c in self.net_classes:
            if any(p.net == net for p in c.promoted):
                return c
        for c in self.net_classes:
            if net in c.nets:
                return c
        return self.default_class()

    def declared_class_of(self, net: str) -> NetClass | None:
        """The class that lists ``net`` (not a promotion), else the default class."""
        for c in self.net_classes:
            if net in c.nets:
                return c
        return self.default_class()

    def promotion_of(self, net: str) -> Promotion | None:
        for c in self.net_classes:
            for p in c.promoted:
                if p.net == net:
                    return p
        return None

    def timing_path(self, name: str) -> TimingPath | None:
        for p in self.timing_paths:
            if p.name == name:
                return p
        return None

    def traced_items(self, prefix: str = SI_PREFIX) -> list[tuple[str, Traced]]:
        """``(id, traced)`` of every traced SI number, in model order (the ids :meth:`lookup` resolves)."""
        out: list[tuple[str, Traced]] = []
        if self.critical_fraction is not None:
            out.append((f"{prefix}.critical_fraction", self.critical_fraction))
        for c in self.net_classes:
            for field in CLASS_UNITS:
                t = getattr(c, field)
                if t is not None:
                    out.append((f"{prefix}.net_classes[{c.name}].{field}", t))
        for p in self.timing_paths:
            for field in ("f_clk_hz", "capture_fraction", *TIMING_TERMS):
                t = getattr(p, field)
                if t is not None:
                    out.append((f"{prefix}.timing_paths[{p.name}].{field}", t))
        return out

    def lookup(self, key: str, prefix: str = SI_PREFIX) -> Traced | None:
        """The traced value with id ``key`` (see :meth:`traced_items`), else ``None``."""
        if not key.startswith(prefix + "."):
            return None
        for k, t in self.traced_items(prefix):
            if k == key:
                return t
        return None


__all__ = [
    "CLASS_UNITS",
    "FACT_REF_RE",
    "SI_PREFIX",
    "TIMING_TERMS",
    "DiffPair",
    "NetClass",
    "Promotion",
    "SIConstraints",
    "TimingPath",
]
