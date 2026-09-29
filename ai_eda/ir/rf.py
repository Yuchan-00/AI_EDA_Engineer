"""RF design content of the IR (``ir.rf``): blocks, fixture networks, the frequency plan, lab items, rail budgets.

Invariant: this is *design content* - what an RF design declares about
itself and what its passive networks must show - never a verdict, and every
number in it is :class:`~ai_eda.ir.provenance.Traced`: a template's
confirmed choice (``model.*`` values included), the user's value, a grounded
fact or a registered calculator's output (``derived``, re-derived by
``calc.recompute`` through the ids :meth:`RFDesign.traced_items` names; such
a value takes its inputs from ``ir.parameters``, the requirements or the
stackup, never from an ``rf.*`` path). The checks that read it judge: the
fixture runner (``spice.rf.<network>[.<state>].<expectation>``, ngspice AC
of a network under confirmed model values - a verdict about that network,
never about a part, the board or the RF performance), the RF checks
(``rf.freq_plan``, ``rf.regulatory_profile``, ``rf.model_grounding``,
``rf.deviation``, ``rf.lab.<id>``, ``block.interface.<net>``,
``power.rail_budget.<rail>`` / ``power.headroom.<regulator>``) and the RF
floorplan placer. ``CircuitIR.rf`` is ``None`` for a design that states no
RF content and is then left out of the design view, so an IR saved before
the field existed keeps its hash.

The models refuse what they cannot mean (a ``ValueError`` naming it):

* :class:`RFPort` - a named connection point of a block or a fixture
  network: ``kind`` ``"port"`` (a real resistance ``z0_ohm`` > 0, required:
  a source through it when driven, a load otherwise), ``"probe"`` (no load
  and no ``z0_ohm``: "S21" to a probe is the voltage ratio 2 V_to / V_s,
  and only its phase or relative level means anything), ``"rail"`` /
  ``"control"`` (``voltage_v`` for the rails' agreement at a block
  interface; in a fixture network an ideal DC source - the state's
  ``port_dc_v``, a rail's ``voltage_v`` otherwise - which is also an ac
  short at that node: a bypassed rail or a bias feed, never driven or read);
  ``frequency_hz`` > 0 when stated.
* :class:`RFState` - one bias / switch state of a fixture network: DC levels
  on its ports (``port_dc_v``) and binding overrides of its members.
* :class:`RFExpectation` - one judged quantity of a network at ``at`` (Hz):
  ``s21_db`` (power-wave S21 between two loaded ports), ``s11_db`` (at the
  drive port itself), ``rel_s21_db`` (S21 at ``at`` relative to S21 at
  ``ref_at``) or ``phase21_deg``. It carries exactly one of ``tol_abs`` (> 0,
  in dB or deg) and ``bound`` (``at_least`` / ``at_most``: PASS when the
  measured value is on the passing side of ``nominal`` -
  :func:`ai_eda.tools.spice.stage.judge`; the runner reads every row at
  ``at`` / ``ref_at`` themselves, from its own single-point analyses, never
  between the points of a sweep).
  ``tol_rel`` is always ``None`` (a relative tolerance on a level in dB or on
  a phase is no tolerance; the field exists so ``judge`` reads one shape for
  an :class:`~ai_eda.ir.simulation.Expectation` and an RF expectation), and
  a one-sided bound cannot claim a requirement yet (``requirement_id``).
* :class:`RFProbe` - the same point without a verdict (recorded only).
* :class:`RFNetwork` - a passive network cut out of the schematic for an AC
  fixture: its ``members`` (refs), fixture ``bindings`` (a member without one
  uses its ``Component.spice``), ``loss_q`` (an inductor listed there is
  simulated with a series R = 2 pi ``q_ref_hz`` L / Q - the only element
  besides the port sources / loads and the DC-path resistors the runner
  adds), ``ports`` (in any order - every expectation and probe names its
  ``drive``; a ``rail`` / ``control`` port is a DC level, never the drive
  or the port read), ``states``, the ac ``sweep`` and the expectations /
  probes. What the runner could not simulate is refused here: a DC level on
  a probe (no element carries it), a ``control`` port without a level in
  every state (the runner takes a control's level only from the state) or
  in a network without states, a ``rail`` without ``voltage_v`` and without
  a level in every state, and a sweep id that starts with
  :data:`POINT_ANALYSIS_PREFIX` (the runner's own point analyses).
* :class:`PlanLine` - one row of the frequency plan: ``margin`` (``f_hz``
  must stay ``min_margin_hz`` away from ``ref_hz``), ``coincidence``
  (``f_hz`` against ``ref_hz``), ``response`` / ``gated`` (their verdict is
  elsewhere: ``points_to`` names the check / lab ids).
* :class:`LabItem` - what only a lab can measure (``rf.lab.<id>``).
* :class:`RailBudget` - a regulator's rail: output voltage, the load current
  range, the rating, the dropout and the path resistance before it.
* :class:`RFBlock` - a functional block: its refs, the signal ``chain``
  order, the shield can (``shield_ref``), the floorplan ``region``
  (:class:`RFRegion`, mm in the board frame of the placements) and its
  interface ``ports``. A ref belongs to at most one block.
* :class:`RFDesign` - all of it, plus ``model_values`` (the ``model.*``
  parameter keys no datasheet or measurement grounds) and ``profile_keys``
  (the regulatory-profile parameter keys: unverified choices, never facts).

Identifiers (network, state, expectation, probe, block, plan-line and lab
ids, port names) are plain identifiers: they become check-id segments and
deck stems, so they carry no ``.``, bracket or space.
"""

from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from ai_eda.ir.provenance import Traced
from ai_eda.ir.simulation import EXPECTATION_BOUNDS, AnalysisSpec, SpiceBinding
from ai_eda.tools.spice.runner import SpiceAnalysis

#: the id prefix of every traced RF number (``rf.networks[lpf].expectations[s21_fc].nominal``), resolved by :meth:`RFDesign.lookup`
RF_PREFIX = "rf"
#: plain identifiers: ids and port names become check-id segments (``spice.rf.<network>.<state>.<exp>``) and deck stems
ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
#: a rail is a net name (``V_SYS``, ``+5V``) that becomes the check id ``power.rail_budget.<rail>``
RAIL_RE = re.compile(r"^[A-Za-z0-9_+\-]+$")
#: what a port is (module docstring)
PORT_KINDS: tuple[str, ...] = ("port", "probe", "rail", "control")
#: the port kinds a fixture network may drive or read: a real resistance or a high-impedance probe (a rail / control port is only a DC level there)
FIXTURE_PORT_KINDS: tuple[str, ...] = ("port", "probe")
#: each judged / probed quantity and its unit
RF_QUANTITIES: dict[str, str] = {"s21_db": "dB", "s11_db": "dB", "rel_s21_db": "dB", "phase21_deg": "deg"}
#: the one-sided bounds (the same as :attr:`ai_eda.ir.simulation.Expectation.bound`'s)
BOUNDS: tuple[str, ...] = EXPECTATION_BOUNDS
#: the kinds of a frequency-plan row
PLAN_KINDS: tuple[str, ...] = ("margin", "response", "gated", "coincidence")
#: the id prefix of the fixture runner's own single-point analyses (``ac lin 1 f f`` at each row's frequency): no sweep of a network may use it
POINT_ANALYSIS_PREFIX = "rf_at_"

PortKind = Literal["port", "probe", "rail", "control"]
Direction = Literal["in", "out", "bidir"]
Quantity = Literal["s21_db", "s11_db", "rel_s21_db", "phase21_deg"]
Bound = Literal["at_least", "at_most"]
PlanKind = Literal["margin", "response", "gated", "coincidence"]


def _check_id(value: str, what: str) -> None:
    if not isinstance(value, str) or not ID_RE.match(value):
        raise ValueError(f"{what} {value!r} must be a plain identifier (a letter, then letters, digits or _): it becomes a check-id segment")


def _check_name(value: str, what: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip() or any(ch.isspace() for ch in value):
        raise ValueError(f"{what} {value!r} must be a non-empty name without spaces")


def _check_number(t: Traced | None, what: str, unit: str | None, *, positive: bool = False, non_negative: bool = False) -> None:
    """``ValueError`` when ``t`` carries another unit than ``unit`` or is not a finite number in range."""
    if t is None:
        return
    if t.unit != unit:
        want = f"unit {unit!r}" if unit is not None else "no unit (a dimensionless number)"
        raise ValueError(f"{what} must carry {want}, got {t.unit!r}")
    v = t.value
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{what} must be a finite number, got {v!r}")
    if positive and not v > 0:
        raise ValueError(f"{what} must be > 0, got {v!r}")
    if non_negative and v < 0:
        raise ValueError(f"{what} must be >= 0, got {v!r}")


def _unique(values: list[str], what: str) -> None:
    seen: set[str] = set()
    for v in values:
        if v in seen:
            raise ValueError(f"{what} {v!r} is used twice")
        seen.add(v)


def _binding_items(prefix: str, bindings: dict[str, SpiceBinding]) -> list[tuple[str, Traced]]:
    out: list[tuple[str, Traced]] = []
    for ref, b in bindings.items():
        if b.value is not None:
            out.append((f"{prefix}.bindings[{ref}].value", b.value))
        for k, t in b.params.items():
            out.append((f"{prefix}.bindings[{ref}].params[{k}]", t))
    return out


# --------------------------------------------------------------------------- ports, regions, states


class RFPort(BaseModel):
    """A named connection point of a block or a fixture network (module docstring)."""

    name: str
    net: str
    reference_net: str = "GND"
    kind: PortKind
    #: the port's reference impedance (a real resistance, ohm, > 0): required on a ``port``, refused elsewhere
    z0_ohm: Traced[float] | None = None
    #: the highest frequency the port carries (Hz, > 0), for ``block.interface`` and the frequency plan
    frequency_hz: Traced[float] | None = None
    #: a rail's (or a control line's) voltage, for ``block.interface``'s agreement check
    voltage_v: Traced[float] | None = None
    direction: Direction = "bidir"

    @model_validator(mode="after")
    def _consistent(self) -> RFPort:
        _check_id(self.name, "port name")
        _check_name(self.net, f"port {self.name}: net")
        _check_name(self.reference_net, f"port {self.name}: reference_net")
        if self.net == self.reference_net:
            raise ValueError(f"port {self.name}: net and reference_net are both {self.net!r}")
        if self.kind == "port" and self.z0_ohm is None:
            raise ValueError(f"port {self.name}: a 'port' needs z0_ohm (the real resistance a source drives through and a load terminates in)")
        if self.kind != "port" and self.z0_ohm is not None:
            raise ValueError(f"port {self.name}: only a 'port' has a reference impedance; a {self.kind!r} takes no z0_ohm" + (" (a probe has no load)" if self.kind == "probe" else ""))
        if self.voltage_v is not None and self.kind not in ("rail", "control"):
            raise ValueError(f"port {self.name}: voltage_v belongs to a 'rail' or 'control' port, not a {self.kind!r}")
        _check_number(self.z0_ohm, f"port {self.name}: z0_ohm", "ohm", positive=True)
        _check_number(self.frequency_hz, f"port {self.name}: frequency_hz", "Hz", positive=True)
        _check_number(self.voltage_v, f"port {self.name}: voltage_v", "V")
        return self


class RFRegion(BaseModel):
    """A block's floorplan rectangle in mm, in the board frame of the placements (Y down; for an outline at origin (0, 0), from its top-left corner)."""

    x: Traced[float]
    y: Traced[float]
    w: Traced[float]
    h: Traced[float]

    @model_validator(mode="after")
    def _consistent(self) -> RFRegion:
        for name in ("x", "y"):
            _check_number(getattr(self, name), f"region {name}", "mm")
        for name in ("w", "h"):
            _check_number(getattr(self, name), f"region {name}", "mm", positive=True)
        return self

    def box(self) -> tuple[float, float, float, float]:
        """``(x0, y0, x1, y1)`` in mm."""
        x, y = float(self.x.value), float(self.y.value)
        return x, y, x + float(self.w.value), y + float(self.h.value)


class RFState(BaseModel):
    """One bias / switch state of a fixture network: DC levels on its ports and binding overrides of its members."""

    id: str
    #: port name -> the DC level (V) the port's source carries in this state
    port_dc_v: dict[str, Traced[float]] = Field(default_factory=dict)
    #: member ref -> the binding this state simulates it with (a PIN diode's R_on in ``tx``, its C_off in ``rx``)
    bindings: dict[str, SpiceBinding] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> RFState:
        _check_id(self.id, "state id")
        for port, t in self.port_dc_v.items():
            _check_number(t, f"state {self.id}: port_dc_v[{port}]", "V")
        return self


# --------------------------------------------------------------------------- expectations and probes


class _RFPoint(BaseModel):
    """The shape an :class:`RFExpectation` and an :class:`RFProbe` share: which quantity, between which ports, at which frequency."""

    id: str
    #: the network state it is measured in (required when the network has states, else ``None``)
    state: str | None = None
    quantity: Quantity
    #: the driven port (a ``port``: a source through its z0)
    drive: str
    #: the port the quantity is read at (``s11_db``: the drive port itself)
    to: str
    at: Traced[float]
    #: the reference frequency of a ``rel_s21_db`` row (S21 at ``at`` relative to S21 at ``ref_at``); ``None`` otherwise
    ref_at: Traced[float] | None = None

    @model_validator(mode="after")
    def _shape(self) -> _RFPoint:
        what = f"{type(self).__name__} {self.id}"
        _check_id(self.id, f"{what}: id")
        if self.state is not None:
            _check_id(self.state, f"{what}: state")
        _check_id(self.drive, f"{what}: drive")
        _check_id(self.to, f"{what}: to")
        if self.quantity == "s11_db" and self.to != self.drive:
            raise ValueError(f"{what}: s11_db is the reflection at the drive port: 'to' must be {self.drive!r}, got {self.to!r}")
        if self.quantity != "s11_db" and self.to == self.drive:
            raise ValueError(f"{what}: {self.quantity} needs two ports (drive and to are both {self.drive!r}; the reflection is s11_db)")
        if (self.quantity == "rel_s21_db") != (self.ref_at is not None):
            raise ValueError(f"{what}: " + ("rel_s21_db needs ref_at (the frequency it is relative to)" if self.ref_at is None else f"ref_at belongs to rel_s21_db, not {self.quantity}"))
        _check_number(self.at, f"{what}: at", "Hz", positive=True)
        _check_number(self.ref_at, f"{what}: ref_at", "Hz", positive=True)
        return self

    @property
    def unit(self) -> str:
        """The quantity's unit (``dB`` / ``deg``)."""
        return RF_QUANTITIES[self.quantity]


class RFExpectation(_RFPoint):
    """A judged quantity of a fixture network (module docstring): exactly one of ``tol_abs`` / ``bound``, never ``tol_rel``."""

    nominal: Traced[float]
    #: the tolerance (> 0, in the quantity's unit); ``None`` when ``bound`` is set
    tol_abs: Traced[float] | None = None
    #: always ``None``: a relative tolerance on a level in dB or on a phase is no tolerance (present so ``judge`` reads one shape)
    tol_rel: Traced[float] | None = None
    #: a one-sided limit: PASS when the measured value (read at ``at`` itself) is on the passing side of ``nominal``
    bound: Bound | None = None
    requirement_id: str | None = None

    @model_validator(mode="after")
    def _verdict(self) -> RFExpectation:
        what = f"RFExpectation {self.id}"
        if self.tol_rel is not None:
            raise ValueError(f"{what}: tol_rel is not allowed - a relative tolerance on a level in dB or on a phase is no tolerance; give tol_abs or a bound")
        if (self.tol_abs is None) == (self.bound is None):
            raise ValueError(f"{what}: give exactly one of tol_abs (a two-sided tolerance) or bound (at_least / at_most), got " + ("both" if self.bound is not None else "neither"))
        if self.bound is not None and self.requirement_id is not None:
            raise ValueError(f"{what}: a one-sided bound cannot claim a requirement yet (requirement_id {self.requirement_id!r}): the requirement checks know only tolerances")
        _check_number(self.nominal, f"{what}: nominal", self.nominal.unit)
        if self.nominal.unit is not None and self.nominal.unit != self.unit:
            raise ValueError(f"{what}: nominal carries unit {self.nominal.unit!r}, {self.quantity} is in {self.unit}")
        if self.tol_abs is not None:
            _check_number(self.tol_abs, f"{what}: tol_abs", self.tol_abs.unit, positive=True)
            if self.tol_abs.unit is not None and self.tol_abs.unit != self.unit:
                raise ValueError(f"{what}: tol_abs carries unit {self.tol_abs.unit!r}, {self.quantity} is in {self.unit}")
        return self


class RFProbe(_RFPoint):
    """A recorded quantity of a fixture network: measured and reported, never judged."""


# --------------------------------------------------------------------------- the fixture network


class RFNetwork(BaseModel):
    """A passive network cut out of the schematic for an AC fixture (module docstring)."""

    id: str
    #: the :class:`RFBlock` it belongs to (``None``: a stage board without blocks)
    block: str | None = None
    #: the component refs that make up the network
    members: list[str]
    #: member ref -> the fixture binding that overrides its ``Component.spice``
    bindings: dict[str, SpiceBinding] = Field(default_factory=dict)
    #: inductor ref -> its unloaded Q (dimensionless, > 0) at ``q_ref_hz``: simulated as a series R = 2 pi q_ref_hz L / Q
    loss_q: dict[str, Traced[float]] = Field(default_factory=dict)
    q_ref_hz: Traced[float] | None = None
    #: the network's ports (``port`` / ``probe`` / ``rail`` / ``control``) in any order: each expectation / probe names its drive
    ports: list[RFPort]
    states: list[RFState] = Field(default_factory=list)
    #: the ac analyses the network is swept with
    sweep: list[AnalysisSpec]
    expectations: list[RFExpectation] = Field(default_factory=list)
    probes: list[RFProbe] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> RFNetwork:
        what = f"network {self.id}"
        _check_id(self.id, "network id")
        if self.block is not None:
            _check_id(self.block, f"{what}: block")
        if not self.members:
            raise ValueError(f"{what}: needs at least one member")
        for ref in self.members:
            _check_name(ref, f"{what}: member")
        _unique(self.members, f"{what}: member")
        members = set(self.members)
        for label, refs in (("bindings", self.bindings), ("loss_q", self.loss_q)):
            outside = sorted(set(refs) - members)
            if outside:
                raise ValueError(f"{what}: {label} names {outside}, which are not members {sorted(members)}")
        for ref, q in self.loss_q.items():
            _check_number(q, f"{what}: loss_q[{ref}]", None, positive=True)
        if self.loss_q and self.q_ref_hz is None:
            raise ValueError(f"{what}: loss_q needs q_ref_hz (the frequency the series R = 2 pi f L / Q is taken at)")
        if self.q_ref_hz is not None and not self.loss_q:
            raise ValueError(f"{what}: q_ref_hz without loss_q means nothing")
        _check_number(self.q_ref_hz, f"{what}: q_ref_hz", "Hz", positive=True)
        if not self.ports:
            raise ValueError(f"{what}: needs ports")
        _unique([p.name for p in self.ports], f"{what}: port name")
        _unique([p.net for p in self.ports], f"{what}: port net")
        if not any(p.kind == "port" for p in self.ports):
            raise ValueError(f"{what}: needs at least one 'port' (a probe cannot be driven)")
        ports = {p.name: p for p in self.ports}
        _unique([s.id for s in self.states], f"{what}: state id")
        for s in self.states:
            outside = sorted(set(s.port_dc_v) - set(ports))
            if outside:
                raise ValueError(f"{what}: state {s.id} sets port_dc_v on {outside}, which are not ports of the network")
            outside = sorted(set(s.bindings) - members)
            if outside:
                raise ValueError(f"{what}: state {s.id} binds {outside}, which are not members {sorted(members)}")
            probes = sorted(name for name in s.port_dc_v if ports[name].kind == "probe")
            if probes:
                raise ValueError(f"{what}: state {s.id} sets port_dc_v on the probe(s) {probes}: a probe has no element to carry a DC level")
        if not self.sweep:
            raise ValueError(f"{what}: needs an ac sweep")
        _unique([a.id for a in self.sweep], f"{what}: sweep id")
        for a in self.sweep:
            _check_id(a.id, f"{what}: sweep id")
            if a.kind != SpiceAnalysis.AC:
                raise ValueError(f"{what}: sweep {a.id} is a {a.kind.value} analysis; a fixture network is swept with ac only")
            if a.id.lower().startswith(POINT_ANALYSIS_PREFIX):
                raise ValueError(f"{what}: sweep id {a.id!r} starts with {POINT_ANALYSIS_PREFIX!r}, the prefix of the fixture runner's own point analyses")
        _unique([e.id for e in (*self.expectations, *self.probes)], f"{what}: expectation / probe id")
        states = {s.id for s in self.states}
        for e in (*self.expectations, *self.probes):
            label = f"{what}: {'expectation' if isinstance(e, RFExpectation) else 'probe'} {e.id}"
            if states and e.state not in states:
                raise ValueError(f"{label}: state must be one of {sorted(states)}, got {e.state!r}")
            if not states and e.state is not None:
                raise ValueError(f"{label}: names state {e.state!r}, but the network has no states")
            for end in (e.drive, e.to):
                if end not in ports:
                    raise ValueError(f"{label}: port {end!r} is not a port of the network {sorted(ports)}")
            if ports[e.drive].kind != "port":
                raise ValueError(f"{label}: drive {e.drive!r} is a {ports[e.drive].kind}; only a 'port' can be driven (a source through its z0)")
            if ports[e.to].kind not in FIXTURE_PORT_KINDS:
                raise ValueError(f"{label}: port {e.to!r} is a {ports[e.to].kind} (an ideal DC source, an ac short); only a 'port' or a 'probe' can be read")
            if e.quantity in ("s21_db", "s11_db") and ports[e.to].kind != "port":
                raise ValueError(f"{label}: {e.quantity} to the probe {e.to!r} is no S-parameter (a probe has no load); use rel_s21_db or phase21_deg")
        # a DC-level port the runner would have no level for (it takes a control's level only from the state, a rail's
        # from the state or its voltage_v): refused here, not discovered as a deck that cannot be built
        for p in self.ports:
            if p.kind == "control" or (p.kind == "rail" and p.voltage_v is None):
                why = "a control port takes its DC level only from the state" if p.kind == "control" else "a rail without voltage_v takes its DC level from the state"
                if not self.states:
                    raise ValueError(f"{what}: {p.kind} port {p.name} needs a DC level, but the network has no states ({why}: give the network a state "
                                     f"with port_dc_v[{p.name}]" + (" or the rail a voltage_v)" if p.kind == "rail" else ")"))
                missing = [s.id for s in self.states if p.name not in s.port_dc_v]
                if missing:
                    raise ValueError(f"{what}: {p.kind} port {p.name} needs a DC level in every state ({why}); state(s) {missing} give none")
        return self

    def port(self, name: str) -> RFPort | None:
        return next((p for p in self.ports if p.name == name), None)

    def state(self, id: str) -> RFState | None:
        return next((s for s in self.states if s.id == id), None)

    def traced_items(self, prefix: str) -> list[tuple[str, Traced]]:
        """``(id, traced)`` of every traced value of the network under ``prefix`` (``rf.networks[<id>]``)."""
        out: list[tuple[str, Traced]] = []
        for p in self.ports:
            out.extend(_port_items(f"{prefix}.ports[{p.name}]", p))
        out.extend(_binding_items(prefix, self.bindings))
        for ref, q in self.loss_q.items():
            out.append((f"{prefix}.loss_q[{ref}]", q))
        if self.q_ref_hz is not None:
            out.append((f"{prefix}.q_ref_hz", self.q_ref_hz))
        for s in self.states:
            here = f"{prefix}.states[{s.id}]"
            for port, t in s.port_dc_v.items():
                out.append((f"{here}.port_dc_v[{port}]", t))
            out.extend(_binding_items(here, s.bindings))
        for a in self.sweep:
            for k, t in a.params.items():
                out.append((f"{prefix}.sweep[{a.id}].params[{k}]", t))
        for e in self.expectations:
            for field in ("at", "ref_at", "nominal", "tol_abs", "tol_rel"):
                t = getattr(e, field)
                if t is not None:
                    out.append((f"{prefix}.expectations[{e.id}].{field}", t))
        for p in self.probes:
            for field in ("at", "ref_at"):
                t = getattr(p, field)
                if t is not None:
                    out.append((f"{prefix}.probes[{p.id}].{field}", t))
        return out


def _port_items(prefix: str, p: RFPort) -> list[tuple[str, Traced]]:
    return [(f"{prefix}.{field}", t) for field in ("z0_ohm", "frequency_hz", "voltage_v") for t in [getattr(p, field)] if t is not None]


# --------------------------------------------------------------------------- plan, lab, rails, blocks


class PlanLine(BaseModel):
    """One row of the frequency plan (module docstring)."""

    id: str
    kind: PlanKind
    f_hz: Traced[float]
    ref_hz: Traced[float] | None = None
    #: the least distance ``f_hz`` must keep from ``ref_hz`` (Hz, >= 0): margin rows only
    min_margin_hz: Traced[float] | None = None
    #: check ids / lab ids that carry this row's verdict (response / gated rows) or its evidence
    points_to: list[str] = Field(default_factory=list)
    note: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> PlanLine:
        what = f"plan line {self.id}"
        _check_id(self.id, "plan line id")
        _check_number(self.f_hz, f"{what}: f_hz", "Hz", positive=True)
        _check_number(self.ref_hz, f"{what}: ref_hz", "Hz", positive=True)
        _check_number(self.min_margin_hz, f"{what}: min_margin_hz", "Hz", non_negative=True)
        if self.kind == "margin" and (self.ref_hz is None or self.min_margin_hz is None):
            raise ValueError(f"{what}: a margin row needs ref_hz and min_margin_hz")
        if self.kind != "margin" and self.min_margin_hz is not None:
            raise ValueError(f"{what}: min_margin_hz belongs to a margin row, not a {self.kind!r} row")
        if self.kind == "coincidence" and self.ref_hz is None:
            raise ValueError(f"{what}: a coincidence row needs ref_hz")
        if self.kind in ("response", "gated") and not self.points_to:
            raise ValueError(f"{what}: a {self.kind} row carries no verdict of its own: points_to must name the checks / lab items that do")
        for target in self.points_to:
            _check_name(target, f"{what}: points_to entry")
        return self


class LabItem(BaseModel):
    """What only a lab can measure: the check ``rf.lab.<id>`` (NOT_VERIFIED without lab evidence)."""

    id: str
    block: str | None = None
    what: str
    instruments: list[str] = Field(default_factory=list)
    reason: str

    @model_validator(mode="after")
    def _consistent(self) -> LabItem:
        _check_id(self.id, "lab item id")
        if self.block is not None:
            _check_id(self.block, f"lab item {self.id}: block")
        if not self.what.strip() or not self.reason.strip():
            raise ValueError(f"lab item {self.id}: 'what' and 'reason' must say something")
        return self


class RailBudget(BaseModel):
    """A regulator's rail: its output, the load current range, the rating, the dropout and the path resistance before it.

    ``i_rating`` / ``dropout_v`` / ``path_r_ohm`` may be unstated (``None``);
    every current and the dropout are datasheet facts, so they stay choices
    until grounded.
    """

    rail: str
    regulator_ref: str
    v_out: Traced[float]
    i_min: Traced[float]
    i_max: Traced[float]
    i_rating: Traced[float] | None = None
    dropout_v: Traced[float] | None = None
    path_r_ohm: Traced[float] | None = None

    @model_validator(mode="after")
    def _consistent(self) -> RailBudget:
        what = f"rail {self.rail}"
        if not isinstance(self.rail, str) or not RAIL_RE.match(self.rail):
            raise ValueError(f"rail {self.rail!r} must be a net name of letters, digits, _ + - (it becomes the check id power.rail_budget.<rail>)")
        _check_name(self.regulator_ref, f"{what}: regulator_ref")
        _check_number(self.v_out, f"{what}: v_out", "V", positive=True)
        _check_number(self.i_min, f"{what}: i_min", "A", non_negative=True)
        _check_number(self.i_max, f"{what}: i_max", "A", non_negative=True)
        if float(self.i_max.value) < float(self.i_min.value):
            raise ValueError(f"{what}: i_max {self.i_max.value!r} A is below i_min {self.i_min.value!r} A")
        _check_number(self.i_rating, f"{what}: i_rating", "A", positive=True)
        _check_number(self.dropout_v, f"{what}: dropout_v", "V", non_negative=True)
        _check_number(self.path_r_ohm, f"{what}: path_r_ohm", "ohm", non_negative=True)
        return self


class RFBlock(BaseModel):
    """A functional block (module docstring): refs, the signal chain order, the shield can, the floorplan region, the interface ports."""

    id: str
    title: str = ""
    refs: list[str] = Field(default_factory=list)
    #: the refs along the signal path, in order (a subset of ``refs``)
    chain: list[str] = Field(default_factory=list)
    #: the shield can over the block (one of ``refs``)
    shield_ref: str | None = None
    region: RFRegion | None = None
    ports: list[RFPort] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> RFBlock:
        what = f"block {self.id}"
        _check_id(self.id, "block id")
        for ref in self.refs:
            _check_name(ref, f"{what}: ref")
        _unique(self.refs, f"{what}: ref")
        _unique(self.chain, f"{what}: chain ref")
        outside = [r for r in self.chain if r not in set(self.refs)]
        if outside:
            raise ValueError(f"{what}: chain names {outside}, which are not refs of the block")
        if self.shield_ref is not None and self.shield_ref not in self.refs:
            raise ValueError(f"{what}: shield_ref {self.shield_ref!r} is not a ref of the block")
        _unique([p.name for p in self.ports], f"{what}: port name")
        return self

    def port(self, name: str) -> RFPort | None:
        return next((p for p in self.ports if p.name == name), None)


# --------------------------------------------------------------------------- the whole


class RFDesign(BaseModel):
    """``ir.rf``: blocks, fixture networks, the frequency plan, lab items, rail budgets, model values and profile keys (module docstring)."""

    blocks: list[RFBlock] = Field(default_factory=list)
    networks: list[RFNetwork] = Field(default_factory=list)
    frequency_plan: list[PlanLine] = Field(default_factory=list)
    lab_items: list[LabItem] = Field(default_factory=list)
    rails: list[RailBudget] = Field(default_factory=list)
    #: the ``model.*`` parameter keys (``ir.parameters``) that no datasheet or measurement grounds
    model_values: list[str] = Field(default_factory=list)
    #: the regulatory-profile parameter keys (``kr447.max_power`` ...): unverified choices, never grounded facts
    profile_keys: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> RFDesign:
        _unique([b.id for b in self.blocks], "block id")
        _unique([n.id for n in self.networks], "network id")
        _unique([p.id for p in self.frequency_plan], "plan line id")
        _unique([x.id for x in self.lab_items], "lab item id")
        _unique([r.rail for r in self.rails], "rail")
        _unique([r.regulator_ref for r in self.rails], "rail regulator_ref")
        owner: dict[str, str] = {}
        for b in self.blocks:
            for ref in b.refs:
                if ref in owner:
                    raise ValueError(f"ref {ref!r} is in two blocks ({owner[ref]}, {b.id})")
                owner[ref] = b.id
        blocks = {b.id for b in self.blocks}
        for label, items in (("network", self.networks), ("lab item", self.lab_items)):
            for x in items:
                if x.block is not None and x.block not in blocks:
                    raise ValueError(f"{label} {x.id}: block {x.block!r} is not a block of the design {sorted(blocks)}")
        for key in self.model_values:
            if not isinstance(key, str) or not key.startswith("model.") or len(key) <= len("model."):
                raise ValueError(f"model value key {key!r} must start with 'model.' (a modelling number no datasheet or measurement grounds)")
        _unique(self.model_values, "model value key")
        for key in self.profile_keys:
            _check_name(key, "profile key")
        _unique(self.profile_keys, "profile key")
        return self

    # --- lookups -------------------------------------------------------------

    def block(self, id: str) -> RFBlock | None:
        return next((b for b in self.blocks if b.id == id), None)

    def network(self, id: str) -> RFNetwork | None:
        return next((n for n in self.networks if n.id == id), None)

    def block_of(self, ref: str) -> RFBlock | None:
        """The block that lists ``ref``, else ``None``."""
        return next((b for b in self.blocks if ref in b.refs), None)

    def traced_items(self, prefix: str = RF_PREFIX) -> list[tuple[str, Traced]]:
        """``(id, traced)`` of every traced value of the RF design, in model order (the ids :meth:`lookup` resolves).

        Binding model cards are text, never a calculator's output, and are not
        listed; every other ``Traced`` is: block regions and ports, network
        ports, bindings' values and params, ``loss_q`` / ``q_ref_hz``, state
        DC levels and bindings, sweep params, expectation and probe numbers,
        the frequency plan and the rail budgets.
        """
        out: list[tuple[str, Traced]] = []
        for b in self.blocks:
            here = f"{prefix}.blocks[{b.id}]"
            if b.region is not None:
                for field in ("x", "y", "w", "h"):
                    out.append((f"{here}.region.{field}", getattr(b.region, field)))
            for p in b.ports:
                out.extend(_port_items(f"{here}.ports[{p.name}]", p))
        for n in self.networks:
            out.extend(n.traced_items(f"{prefix}.networks[{n.id}]"))
        for line in self.frequency_plan:
            for field in ("f_hz", "ref_hz", "min_margin_hz"):
                t = getattr(line, field)
                if t is not None:
                    out.append((f"{prefix}.frequency_plan[{line.id}].{field}", t))
        for r in self.rails:
            for field in ("v_out", "i_min", "i_max", "i_rating", "dropout_v", "path_r_ohm"):
                t = getattr(r, field)
                if t is not None:
                    out.append((f"{prefix}.rails[{r.rail}].{field}", t))
        return out

    def lookup(self, key: str, prefix: str = RF_PREFIX) -> Traced | None:
        """The traced value with id ``key`` (see :meth:`traced_items`), else ``None``."""
        if not key.startswith(prefix + "."):
            return None
        for k, t in self.traced_items(prefix):
            if k == key:
                return t
        return None


__all__ = [
    "BOUNDS",
    "FIXTURE_PORT_KINDS",
    "ID_RE",
    "LabItem",
    "PLAN_KINDS",
    "POINT_ANALYSIS_PREFIX",
    "PORT_KINDS",
    "PlanLine",
    "RAIL_RE",
    "RFBlock",
    "RFDesign",
    "RFExpectation",
    "RFNetwork",
    "RFPort",
    "RFProbe",
    "RFRegion",
    "RFState",
    "RF_PREFIX",
    "RF_QUANTITIES",
    "RailBudget",
]
