"""The block builder API of the RF templates: a block builds its parts, nets, ports, fixtures and simulation once, and a board names it.

Invariant: a block is a pure function of the confirmed inputs, the library
on disk and its prefix - ``Block.build(ctx, prefix)`` returns a
:class:`BlockResult` whose every name follows from ``prefix`` alone, so the
same block builds the same bench board and, re-based, the same part of the
transceiver, byte for byte. A block never writes the IR; the template turns
its result into ``DesignChange`` proposals like every other template, and a
PASS never carries over from a stage board to the composition (the composed
IR hashes differently and every check runs again on it).

Naming (:class:`BlockPrefix`): a block writes block-local references
(letters + a number 1..99: ``R6``, ``C_T1``, ``SH1``) and net names; the
prefix adds ``ref_base`` to every number (``R6`` + 100 -> ``R106``: power
1xx, ptt 2xx, tx_audio 3xx, rx_audio 4xx, if_backend 5xx ... as the kr447
design numbers them) and ``net_prefix`` to every *internal* net. The block's
**interface nets** (the names the composition keeps: ``V_SYS``, ``RX_5V``,
``PM_DRIVE``, ``DISC_OUT`` ...) and the ground net ``GND`` are never renamed,
so two blocks meet exactly where they name the same interface net and
nowhere else (:func:`merge_results` refuses an internal net two blocks share).
Renaming reaches everything that names a part or a net: pin references,
SPICE vectors (``v(NET)`` / ``i(REF)``), stimuli, RF ports, fixture networks
(members, bindings, loss Q, states), the chain, the shield, rail budgets,
net classes and constraint targets (a part or a net of the block; ``*`` and
a block id stay - a constraint's ``description`` is free text and is not
rewritten). Check ids do not change: expectation, analysis, stimulus,
network and constraint ids are the block author's and must be unique on a
board (the RF checks read ``spice.pm_couple_1k`` / ``pm_mod*`` by those ids).

Structure (:meth:`BlockResult.check`, run by :meth:`Block.build`): unique
references, nets naming existing pins, no pin in two nets, **stacked pins in
one net** (the schematic compiler refuses them apart), every pin wired or a
library ``no_connect`` / a pin the block marked open on purpose
(:meth:`BlockBuilder.leave_open`: the IR's ``no_connect`` design decision),
fixture members / ports / vectors naming parts and nets the block has,
constraint targets that are ``*``, the block, one part or one net. A
problem refuses the block (:class:`~ai_eda.design.library_parts.TemplateRefusal`),
never a partial board.

SPICE (:func:`exclude_floating`, the generalisation of the
``atmega128_devboard`` template's hand-made list): an included two-terminal
element one of whose nodes no other included element and no stimulus
touches carries no current at any frequency, so excluding it changes nothing
the netlist computes - and left in, a capacitor there is a DC-floating node
(ngspice-42: singular matrix at the operating point). It is excluded with that
reason, repeatedly until nothing changes (a dead branch R - C hanging off an
excluded IC goes as a whole); a transistor or subcircuit with such a node is
reported (``unresolved``), never excluded silently, because excluding it
would change the circuit. On the ``atmega128_devboard`` board it finds the
template's four hand-made exclusions (C1, C7, C9, C10) and also the ~PEN
pull-up R3, whose node reaches only the MCU - kept by that template, harmless
there, and just as dead. It removes dead ends only: a DC-floating island
without one (two capacitors in series, an op-amp input fed only through a
capacitor) is not detected here - ngspice's operating point refuses it and
the block must give that node a DC path.

The fixture types (``RFPort``, ``RFNetwork``, ``RailBudget``, ``PlanLine``,
``LabItem``) are :mod:`ai_eda.ir.rf`'s; this module reads them by their
contract field names only (kr447 design §5, critic2 contracts) and never
constructs them.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ai_eda.ir import (
    AnalysisSpec,
    CircuitIR,
    Component,
    Constraint,
    Expectation,
    Net,
    NetKind,
    PinElectricalType,
    PinRef,
    Provenance,
    SpiceBinding,
    Stimulus,
    Traced,
)
from ai_eda.ir.provenance import design_data
from ai_eda.ir.simulation import TWO_TERMINAL_DEVICES
from ai_eda.tools.kicad.library import KicadLibrary

from ai_eda.design.base import Choice, choice_provenance, structural_provenance
from ai_eda.design.inputs import DesignInput
from ai_eda.design.library_parts import TemplateRefusal
from ai_eda.design.rf.models import ModelCard, ModelValue, card_choice, model_choice
from ai_eda.design.rf.parts import PlacedPart, instantiate

if TYPE_CHECKING:  # the fixture IR types (kr447 wave 1, part P1)
    from ai_eda.ir.rf import LabItem, PlanLine, RailBudget, RFNetwork, RFPort

#: the one ground net: never renamed, never floating
GROUND_NET = "GND"
#: a block-local reference: letters (``_`` allowed after the first) and a number 1..99 when re-based
_LOCAL_REF = re.compile(r"^([A-Za-z][A-Za-z_]*?)([1-9][0-9]*)$")
_NET_PREFIX = re.compile(r"^(?:[A-Za-z][A-Za-z0-9_]*)?$")
#: the SPICE vector grammar of :class:`~ai_eda.ir.Expectation` (``v(NET)``, ``vp(NET)``, ``i(REF or STIMULUS)`` ...)
_VECTOR = re.compile(r"^(\s*)([vViI])([pPrRiI]?)(\s*\(\s*)([^()\s]+)(\s*\)\s*)$")


def _key(name: str) -> str:
    return name.lower()


@dataclass(frozen=True)
class BlockPrefix:
    """How one block instance is named on a board: ``ref_base`` (a multiple of 100) added to every local number, ``net_prefix`` before every internal net."""

    ref_base: int = 0
    net_prefix: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.ref_base, bool) or not isinstance(self.ref_base, int) or self.ref_base < 0 or self.ref_base % 100:
            raise ValueError(f"ref_base must be a non-negative multiple of 100, got {self.ref_base!r}")
        if not _NET_PREFIX.match(self.net_prefix):
            raise ValueError(f"net_prefix {self.net_prefix!r} must be empty or an identifier (it becomes part of SPICE node names)")

    def ref(self, local: str) -> str:
        """``R6`` -> ``R106`` for ``ref_base`` 100; unchanged for 0. Refuses a local reference that cannot be re-based."""
        if self.ref_base == 0:
            return local
        m = _LOCAL_REF.match(local)
        if m is None or int(m.group(2)) > 99:
            raise TemplateRefusal(f"block reference {local!r} is not letters + a number 1..99, so it cannot be re-based to {self.ref_base}")
        return f"{m.group(1)}{self.ref_base + int(m.group(2))}"

    def net(self, local: str, interface: frozenset[str] | set[str] = frozenset()) -> str:
        """An internal net gets ``net_prefix``; an interface net and ``GND`` keep their names."""
        if not self.net_prefix or local == GROUND_NET or local in interface:
            return local
        return f"{self.net_prefix}{local}"


@dataclass
class BlockContext:
    """What a block reads: the IR (requirements, parameters), the library on disk, the template id and whether the table is confirmed."""

    ir: CircuitIR
    library: KicadLibrary
    template_id: str
    confirmed: bool
    inputs: dict[str, DesignInput] = field(default_factory=dict)
    #: parameters an earlier block of the same board wrote (read-only for this block)
    shared: dict[str, Traced] = field(default_factory=dict)

    def provenance(self, note: str) -> Provenance:
        """Provenance of a structural decision of the template (a part, a net, an analysis)."""
        return structural_provenance(self.template_id, note)

    def requirement_id(self, key: str) -> str | None:
        inp = self.inputs.get(key)
        return None if inp is None else inp.requirement.id


@dataclass
class BlockResult:
    """Everything one block contributes to a board (see the module docstring); the fixture objects are :mod:`ai_eda.ir.rf` types."""

    block_id: str
    title: str = ""
    components: list[Component] = field(default_factory=list)
    nets: list[Net] = field(default_factory=list)
    ports: list[RFPort] = field(default_factory=list)
    networks: list[RFNetwork] = field(default_factory=list)
    #: the signal chain in order (references), for the floorplan
    chain: list[str] = field(default_factory=list)
    choices: list[Choice] = field(default_factory=list)
    #: every parameter the block writes (choices, model values, calculator outputs, copied inputs)
    params: dict[str, Traced] = field(default_factory=dict)
    #: the calculator outputs among ``params`` (key, traced), for the table
    computed: list[tuple[str, Traced]] = field(default_factory=list)
    expectations: list[Expectation] = field(default_factory=list)
    stimuli: list[Stimulus] = field(default_factory=list)
    analyses: list[AnalysisSpec] = field(default_factory=list)
    interface_nets: frozenset[str] = frozenset()
    shield_ref: str | None = None
    placed: dict[str, PlacedPart] = field(default_factory=dict)
    constraints: list[Constraint] = field(default_factory=list)
    #: net class name -> member nets (the template declares the classes; the block names its nets)
    net_classes: dict[str, list[str]] = field(default_factory=dict)
    #: the ``model.*`` keys (values and cards) the block uses: ``ir.rf.model_values``
    model_keys: list[str] = field(default_factory=list)
    lab_items: list[LabItem] = field(default_factory=list)
    plan_lines: list[PlanLine] = field(default_factory=list)
    rails: list[RailBudget] = field(default_factory=list)
    #: pins the block left open on purpose: (ref, pin) -> reason
    open_pins: dict[tuple[str, str], str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def check(self) -> list[str]:
        """Structural problems (see the module docstring); empty when the block may be used."""
        problems: list[str] = []
        by_ref: dict[str, Component] = {}
        for c in self.components:
            if _key(c.ref) in {_key(r) for r in by_ref}:
                problems.append(f"reference {c.ref!r} is used twice (references are case-insensitive in ngspice)")
            by_ref[c.ref] = c
        names: set[str] = set()
        net_of: dict[tuple[str, str], str] = {}
        for n in self.nets:
            if _key(n.name) in names:
                problems.append(f"net {n.name!r} is declared twice")
            names.add(_key(n.name))
            for p in n.pins:
                c = by_ref.get(p.component_ref)
                if c is None:
                    problems.append(f"net {n.name!r} names unknown part {p.component_ref!r}")
                    continue
                if c.pin(p.pin_number) is None:
                    problems.append(f"net {n.name!r} names {p.component_ref}.{p.pin_number}, which is no pin of {c.ref}")
                    continue
                key = (p.component_ref, p.pin_number)
                if key in net_of and net_of[key] != n.name:
                    problems.append(f"pin {p.component_ref}.{p.pin_number} is in nets {net_of[key]!r} and {n.name!r}")
                net_of[key] = n.name
        for ref, placed in self.placed.items():
            if ref not in by_ref:
                problems.append(f"placed part {ref!r} is not among the components")
            for group in placed.stacks:
                nets = {net_of.get((ref, n)) for n in group}
                if len(nets) > 1:
                    problems.append(f"{ref} pins {list(group)} are stacked in the library (KiCad connects them) but sit in nets {sorted(str(x) for x in nets)}")
        for c in self.components:
            for pin in c.pins:
                key = (c.ref, pin.number)
                if key in net_of and key in self.open_pins:
                    problems.append(f"{c.ref}.{pin.number} is marked open ({self.open_pins[key]}) but sits in net {net_of[key]!r}")
                if key in net_of or pin.electrical_type is PinElectricalType.NO_CONNECT or key in self.open_pins:
                    continue
                problems.append(f"{c.ref}.{pin.number} ({pin.name or 'unnamed'}) is in no net and not marked open")
        net_names = {n.name for n in self.nets} | {GROUND_NET}
        for s in self.stimuli:
            for name in (s.net, s.reference_net):
                if name not in net_names:
                    problems.append(f"stimulus {s.id} names net {name!r}, which the block does not have")
        for e in self.expectations:
            for vec in (e.vector, e.reference_vector):
                if vec is None:
                    continue
                m = _VECTOR.match(vec)
                if m is None:
                    problems.append(f"expectation {e.id}: {vec!r} is not a vector of the v(NET) / i(REF) grammar")
                elif m.group(2).lower() == "v" and m.group(5) not in net_names:
                    problems.append(f"expectation {e.id} reads {vec!r}, but the block has no net {m.group(5)!r}")
        for ref in [*self.chain, *([self.shield_ref] if self.shield_ref else [])]:
            if ref not in by_ref:
                problems.append(f"chain / shield names unknown part {ref!r}")
        for p in self.ports:
            for name in (getattr(p, "net", None), getattr(p, "reference_net", None)):
                if name is not None and name not in net_names:
                    problems.append(f"port {getattr(p, 'name', '?')} names net {name!r}, which the block does not have")
        seen_ids: set[str] = set()
        for c in self.constraints:  # a target the renaming can follow: '*', the block, one part or one net - never a name that is both
            if c.id in seen_ids:
                problems.append(f"constraint id {c.id!r} is used twice")
            seen_ids.add(c.id)
            is_ref, is_net = c.target in by_ref, c.target in net_names
            if c.target not in ("*", self.block_id) and not (is_ref or is_net):
                problems.append(f"constraint {c.id} targets {c.target!r}, which is neither '*', the block {self.block_id!r}, a part nor a net of it")
            elif is_ref and is_net:
                problems.append(f"constraint {c.id} targets {c.target!r}, which is both a part and a net of the block (the renaming could not tell which)")
        for nw in self.networks:
            refs = [*getattr(nw, "members", []), *getattr(nw, "bindings", {}), *getattr(nw, "loss_q", {})]
            refs += [r for s in getattr(nw, "states", []) for r in getattr(s, "bindings", {})]
            for r in refs:
                if r not in by_ref:
                    problems.append(f"fixture network {getattr(nw, 'id', '?')} names unknown part {r!r}")
            for p in getattr(nw, "ports", []):
                for name in (getattr(p, "net", None), getattr(p, "reference_net", None)):
                    if name is not None and name not in net_names:
                        problems.append(f"fixture network {getattr(nw, 'id', '?')} port {getattr(p, 'name', '?')} names net {name!r}, which the block does not have")
        return problems


# --------------------------------------------------------------------------- renaming


def rename_vector(vector: str, net: Any, ref: dict[str, str]) -> str:
    """``v(NET)`` / ``vp(NET)`` ... through ``net`` (a callable), ``i(X)`` through ``ref`` when X is a part (a stimulus id stays)."""
    m = _VECTOR.match(vector)
    if m is None:
        return vector
    kind, name = m.group(2), m.group(5)
    new = net(name) if kind.lower() == "v" else ref.get(name, name)
    return f"{m.group(1)}{kind}{m.group(3)}{m.group(4)}{new}{m.group(6)}"


def _renamed_port(port: Any, net: Any) -> Any:
    update = {k: net(getattr(port, k)) for k in ("net", "reference_net") if getattr(port, k, None) is not None}
    return port.model_copy(update=update) if update else port


def _renamed_network(nw: Any, ref: Any, net: Any) -> Any:
    update: dict[str, Any] = {}
    if hasattr(nw, "members"):
        update["members"] = [ref(r) for r in nw.members]
    for name in ("bindings", "loss_q"):
        if hasattr(nw, name):
            update[name] = {ref(k): v for k, v in getattr(nw, name).items()}
    if hasattr(nw, "ports"):
        update["ports"] = [_renamed_port(p, net) for p in nw.ports]
    if hasattr(nw, "states"):
        update["states"] = [s.model_copy(update={"bindings": {ref(k): v for k, v in s.bindings.items()}}) if hasattr(s, "bindings") else s for s in nw.states]
    return nw.model_copy(update=update)


def apply_prefix(result: BlockResult, prefix: BlockPrefix) -> BlockResult:
    """``result`` with every reference re-based and every internal net prefixed (see the module docstring); ``result`` itself is not changed."""
    if prefix == BlockPrefix():
        return result
    keep = result.interface_nets
    ref_map = {c.ref: prefix.ref(c.ref) for c in result.components}
    targets = list(ref_map.values())
    if len({_key(r) for r in targets}) != len(targets):
        raise TemplateRefusal(f"block {result.block_id}: re-basing to {prefix.ref_base} makes two references equal")

    def ref(r: str) -> str:
        return ref_map.get(r, prefix.ref(r))

    def net(n: str) -> str:
        return prefix.net(n, keep)

    block_nets = {n.name for n in result.nets} | {GROUND_NET}  # the names before renaming

    def target(t: str) -> str:
        """A constraint target: a part is re-based, a net of the block renamed (interface nets and GND keep theirs), '*' / a block id kept."""
        return ref_map[t] if t in ref_map else net(t) if t in block_nets else t

    components = [c.model_copy(update={"ref": ref(c.ref)}) for c in result.components]
    by_new = {c.ref: c for c in components}
    nets = [
        n.model_copy(update={"name": net(n.name), "pins": [PinRef(component_ref=ref(p.component_ref), pin_number=p.pin_number) for p in n.pins]})
        for n in result.nets
    ]
    stimuli = [s.model_copy(update={"net": net(s.net), "reference_net": net(s.reference_net)}) for s in result.stimuli]
    expectations = [
        e.model_copy(update={"vector": rename_vector(e.vector, net, ref_map),
                             "reference_vector": None if e.reference_vector is None else rename_vector(e.reference_vector, net, ref_map)})
        for e in result.expectations
    ]
    rails = []
    for r in result.rails:
        update = {}
        if getattr(r, "rail", None) is not None:
            update["rail"] = net(r.rail)
        if getattr(r, "regulator_ref", None) is not None:
            update["regulator_ref"] = ref(r.regulator_ref)
        rails.append(r.model_copy(update=update) if update else r)
    return replace(
        result,
        components=components,
        nets=nets,
        ports=[_renamed_port(p, net) for p in result.ports],
        networks=[_renamed_network(nw, ref, net) for nw in result.networks],
        chain=[ref(r) for r in result.chain],
        stimuli=stimuli,
        expectations=expectations,
        shield_ref=None if result.shield_ref is None else ref(result.shield_ref),
        placed={ref(r): replace(p, component=by_new[ref(r)]) for r, p in result.placed.items()},
        net_classes={cls: [net(n) for n in members] for cls, members in result.net_classes.items()},
        rails=rails,
        open_pins={(ref(r), pin): why for (r, pin), why in result.open_pins.items()},
        constraints=[c.model_copy(update={"target": target(c.target)}) for c in result.constraints],
    )


# --------------------------------------------------------------------------- composition


def _same(a: Any, b: Any) -> bool:
    """Equal as design content (the design view: a provenance's wall-clock ``created_at`` is no difference)."""
    if isinstance(a, BaseModel) and isinstance(b, BaseModel):
        return type(a) is type(b) and design_data(a) == design_data(b)
    return a == b


def merge_results(results: list[BlockResult], block_id: str = "board") -> BlockResult:
    """One board from several (already prefixed) blocks: interface nets joined by name, everything else unique or identical.

    Refuses a reference used twice, an internal net two blocks share (an
    interface net only joins blocks that both declare it), two different
    values under one parameter / stimulus / analysis id, two different
    confirmation-table rows (a choice or a model card) or computed values
    under one key (identical ones merge), and a repeated expectation,
    fixture-network or constraint id.
    """
    out = BlockResult(block_id=block_id, title=" + ".join(r.title or r.block_id for r in results))
    owners: dict[str, str] = {}
    nets: dict[str, Net] = {}
    net_owner: dict[str, BlockResult] = {}
    for r in results:
        for c in r.components:
            if _key(c.ref) in owners:
                raise TemplateRefusal(f"reference {c.ref!r} is used by blocks {owners[_key(c.ref)]!r} and {r.block_id!r}")
            owners[_key(c.ref)] = r.block_id
            out.components.append(c)
        for n in r.nets:
            first = nets.get(n.name)
            if first is None:
                nets[n.name] = n
                net_owner[n.name] = r
                continue
            other = net_owner[n.name]
            if n.name != GROUND_NET and not (n.name in r.interface_nets and n.name in other.interface_nets):
                raise TemplateRefusal(f"net {n.name!r} of block {r.block_id!r} is internal to it or to block {other.block_id!r}: prefix the blocks' internal nets")
            if first.kind != n.kind:
                raise TemplateRefusal(f"interface net {n.name!r} is {first.kind.value} in block {other.block_id!r} and {n.kind.value} in {r.block_id!r}")
            pins = list(first.pins) + [p for p in n.pins if p not in first.pins]
            serves = list(first.serves_requirements) + [s for s in n.serves_requirements if s not in first.serves_requirements]
            nets[n.name] = first.model_copy(update={"pins": pins, "serves_requirements": serves})
        for key, t in r.params.items():
            if key in out.params and not _same(out.params[key], t):
                raise TemplateRefusal(f"parameter {key!r} has different values in two blocks (block {r.block_id!r})")
            out.params[key] = t
        for ch in r.choices:  # a row is merged only when identical: a different row under one key would hide one of them from the table
            have = next((c for c in out.choices if c.key == ch.key), None)
            if have is None:
                out.choices.append(ch)
            elif have != ch:
                raise TemplateRefusal(f"table row {ch.key!r} differs between blocks (block {r.block_id!r}: {ch.description}; before: {have.description}): "
                                      "give each distinct choice or card its own key")
        for key, t in r.computed:
            have_t = next((x for k, x in out.computed if k == key), None)
            if have_t is None:
                out.computed.append((key, t))
            elif not _same(have_t, t):
                raise TemplateRefusal(f"computed value {key!r} differs between blocks (block {r.block_id!r})")
        for attr, what in (("stimuli", "stimulus"), ("analyses", "analysis")):
            have = {x.id: x for x in getattr(out, attr)}
            for x in getattr(r, attr):
                if x.id in have:
                    if not _same(have[x.id], x):
                        raise TemplateRefusal(f"{what} {x.id!r} differs between blocks (block {r.block_id!r})")
                    continue
                getattr(out, attr).append(x)
                have[x.id] = x
        for attr, what in (("expectations", "expectation"), ("networks", "fixture network"), ("constraints", "constraint")):
            ids = {x.id for x in getattr(out, attr)}
            for x in getattr(r, attr):
                if x.id in ids:
                    raise TemplateRefusal(f"{what} id {x.id!r} is used by two blocks (block {r.block_id!r}); check ids must be unique on a board")
                ids.add(x.id)
                getattr(out, attr).append(x)
        out.ports += r.ports
        out.chain += r.chain
        out.placed.update(r.placed)
        for cls, members in r.net_classes.items():
            merged = out.net_classes.setdefault(cls, [])
            merged += [m for m in members if m not in merged]
        out.model_keys = sorted(set(out.model_keys) | set(r.model_keys))
        out.lab_items += r.lab_items
        out.plan_lines += r.plan_lines
        out.rails += r.rails
        out.open_pins.update(r.open_pins)
        out.interface_nets = out.interface_nets | r.interface_nets
        out.notes += r.notes
    out.nets = list(nets.values())
    return out


# --------------------------------------------------------------------------- SPICE: floating nodes


@dataclass(frozen=True)
class Floating:
    """A part with a node no other included element or stimulus touches, and the sentence that says so."""

    ref: str
    net: str
    reason: str


@dataclass
class FloatingReport:
    excluded: list[Floating] = field(default_factory=list)
    #: transistors / subcircuits with such a node: not excluded (that would change the circuit) - fix the circuit or refuse
    unresolved: list[Floating] = field(default_factory=list)


def exclude_floating(components: list[Component], nets: list[Net], stimuli: list[Stimulus], template_id: str) -> tuple[list[Component], FloatingReport]:
    """``components`` with every dangling two-terminal element excluded (repeatedly), and the report (see the module docstring)."""
    ground = {n.name for n in nets if n.kind == NetKind.GROUND}
    net_of = {(p.component_ref, p.pin_number): n.name for n in nets for p in n.pins}
    members: dict[str, list[str]] = {n.name: sorted({p.component_ref for p in n.pins}) for n in nets}
    live: dict[str, int] = {n.name: 0 for n in nets}
    for s in stimuli:
        for name in (s.net, s.reference_net):
            live[name] = live.get(name, 0) + 1

    def element_pins(c: Component) -> list[str]:
        b = c.spice
        if b is None or b.exclude or b.device is None:
            return []
        return [p for p in b.pin_order if (c.ref, p) in net_of]

    included = {c.ref: c for c in components if element_pins(c)}
    for c in included.values():
        for p in element_pins(c):
            live[net_of[(c.ref, p)]] += 1
    report = FloatingReport()
    excluded: dict[str, Floating] = {}

    def dangling(c: Component) -> str | None:
        own: dict[str, int] = {}
        for p in element_pins(c):
            own[net_of[(c.ref, p)]] = own.get(net_of[(c.ref, p)], 0) + 1
        for name in sorted(own):
            if name not in ground and live[name] - own[name] == 0:
                return name
        return None

    changed = True
    while changed:
        changed = False
        for ref in sorted(included):
            c = included[ref]
            if ref in excluded or c.spice is None or c.spice.device not in TWO_TERMINAL_DEVICES:
                continue
            name = dangling(c)
            if name is None:
                continue
            others = [r for r in members.get(name, []) if r != ref]
            reach = f"reaches only parts outside the netlist ({', '.join(others)})" if others else "reaches no other part"
            reason = (
                f"its node {name} {reach} and no stimulus: the part carries no current at any frequency, so excluding it changes nothing the netlist "
                f"computes (left in, a capacitor there is a DC-floating node - ngspice-42: singular matrix at the operating point); structural only"
            )
            excluded[ref] = Floating(ref, name, reason)
            for p in element_pins(c):
                live[net_of[(ref, p)]] -= 1
            changed = True
    for ref in sorted(included):
        c = included[ref]
        if ref in excluded or c.spice is None or c.spice.device in TWO_TERMINAL_DEVICES:
            continue
        name = dangling(c)
        if name is not None:
            report.unresolved.append(Floating(ref, name, f"{ref} ({c.spice.device.value}) has node {name}, which no other element or stimulus touches; "
                                                         f"a multi-terminal part is not excluded automatically (that would change the circuit)"))
    report.excluded = [excluded[r] for r in sorted(excluded)]
    out: list[Component] = []
    for c in components:
        f = excluded.get(c.ref)
        if f is None:
            out.append(c)
            continue
        binding = SpiceBinding(exclude=True, exclude_reason=f.reason, provenance=structural_provenance(template_id, f"{c.ref} excluded from the netlist: floating node {f.net}"))
        out.append(c.model_copy(update={"spice": binding}))
    return out, report


# --------------------------------------------------------------------------- builders


class BlockBuilder:
    """Accumulates one block's :class:`BlockResult`: parts from the parts table, nets by pin function, choices, model values and cards."""

    def __init__(self, ctx: BlockContext, block_id: str, title: str = "", interface_nets: Iterable[str] = ()) -> None:
        self.ctx = ctx
        self.result = BlockResult(block_id=block_id, title=title, interface_nets=frozenset(interface_nets))

    # -- numbers
    def _param(self, key: str, traced: Traced) -> Traced:
        if key in self.result.params:
            raise TemplateRefusal(f"block {self.result.block_id}: parameter {key!r} is written twice")
        self.result.params[key] = traced
        return traced

    def choice(self, key: str, value: Any, unit: str | None, description: str) -> Traced:
        """A free design choice: a table row and a parameter (``assumption`` until the table is confirmed)."""
        prov = choice_provenance(self.ctx.template_id, f"{key} = {value!r}{' ' + unit if unit else ''}: {description}", self.ctx.confirmed)
        self.result.choices.append(Choice(key, description, value, unit))
        return self._param(key, Traced(value=value, unit=unit, provenance=prov))

    def model(self, value: str | ModelValue) -> Traced:
        """A model value (:mod:`ai_eda.design.rf.models`): a table row, a parameter and a ``model_keys`` entry."""
        ch, traced = model_choice(self.ctx.template_id, value, self.ctx.confirmed)
        self.result.choices.append(ch)
        if ch.key not in self.result.model_keys:
            self.result.model_keys.append(ch.key)
        return self._param(ch.key, traced)

    def card(self, card: ModelCard) -> Traced[str]:
        """A model card: a table row and the ``Traced`` text a binding carries (not a parameter), listed in ``model_keys``.

        The same card used again adds no second row; a *different* card under
        a key that already holds a row is refused - its text would become the
        user's on ``confirm_design=yes`` without ever being shown in the table.
        """
        ch, traced = card_choice(self.ctx.template_id, card, self.ctx.confirmed)
        have = next((c for c in self.result.choices if c.key == ch.key), None)
        if have is None:
            self.result.choices.append(ch)
        elif have != ch:
            raise TemplateRefusal(f"block {self.result.block_id}: the table row {ch.key!r} already shows another card ({have.description}); "
                                  f"{card.name} needs its own key (a card's text reaches the user only through its own row)")
        if card.key not in self.result.model_keys:
            self.result.model_keys.append(card.key)
        return traced

    def computed(self, key: str, traced: Traced) -> Traced:
        """A calculator output (``derived`` provenance): a parameter and a table row of the computed section."""
        self.result.computed.append((key, traced))
        return self._param(key, traced)

    # -- parts and nets
    def part(self, key: str, ref: str, value: str, description: str, serves: Iterable[str] = ()) -> PlacedPart:
        placed = instantiate(self.ctx.library, key, ref, value, description, self.ctx.provenance(f"{ref}: {description}"), list(serves))
        if placed.ref in self.result.placed:
            raise TemplateRefusal(f"block {self.result.block_id}: reference {ref!r} is placed twice")
        self.result.placed[placed.ref] = placed
        self.result.components.append(placed.component)
        return placed

    def net(self, name: str, kind: NetKind, members: Iterable[tuple[str, str]], note: str = "", serves: Iterable[str] = ()) -> Net:
        """A net from ``(ref, pin number)`` members (:meth:`PlacedPart.at` gives every pin of a function, stacked ones together)."""
        pins = [PinRef(component_ref=r, pin_number=p) for r, p in members]
        n = Net(name=name, kind=kind, pins=pins, provenance=self.ctx.provenance(note or f"{name} net"), serves_requirements=list(serves))
        self.result.nets.append(n)
        return n

    def leave_open(self, ref: str, function: str, reason: str) -> None:
        """Leave every pin of ``function`` unconnected on purpose: the IR marks it ``no_connect`` (a design decision the compilers accept)."""
        placed = self.result.placed[ref]
        numbers = set(placed.pins(function))
        pins = [p.model_copy(update={"electrical_type": PinElectricalType.NO_CONNECT}) if p.number in numbers else p for p in placed.component.pins]
        component = placed.component.model_copy(update={"pins": pins})
        self.result.components = [component if c.ref == ref else c for c in self.result.components]
        self.result.placed[ref] = replace(placed, component=component)
        for n in sorted(numbers):
            self.result.open_pins[(ref, n)] = reason

    def bind(self, ref: str, binding: SpiceBinding) -> None:
        """Set a part's SPICE binding (the parts table says how each part appears; the block decides)."""
        placed = self.result.placed[ref]
        component = placed.component.model_copy(update={"spice": binding})
        self.result.components = [component if c.ref == ref else c for c in self.result.components]
        self.result.placed[ref] = replace(placed, component=component)

    def done(self) -> BlockResult:
        problems = self.result.check()
        if problems:
            raise TemplateRefusal(f"block {self.result.block_id}: " + "; ".join(problems))
        return self.result


class Block(ABC):
    """One reusable block of the RF templates: ``build_local`` with block-local names, ``build`` checked and prefixed."""

    id: str
    title: str
    #: the nets the composition keeps (never prefixed)
    interface_nets: tuple[str, ...] = ()

    @abstractmethod
    def build_local(self, ctx: BlockContext) -> BlockResult: ...

    def build(self, ctx: BlockContext, prefix: BlockPrefix = BlockPrefix()) -> BlockResult:
        result = self.build_local(ctx)
        result.interface_nets = result.interface_nets | frozenset(self.interface_nets)
        problems = result.check()
        if problems:
            raise TemplateRefusal(f"block {self.id}: " + "; ".join(problems))
        return apply_prefix(result, prefix)


__all__ = [
    "GROUND_NET",
    "Block",
    "BlockBuilder",
    "BlockContext",
    "BlockPrefix",
    "BlockResult",
    "Floating",
    "FloatingReport",
    "apply_prefix",
    "exclude_floating",
    "merge_results",
    "rename_vector",
]
