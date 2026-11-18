"""Opt-in functional intent resolution; no mutation and no layout dispatch.

Absent blocks become connected components of the non-rail graph. This is a
connectivity grouping, not a claim about circuit function. Isolated parts stay
as singleton blocks. Shared rails never establish a support-to-pin assignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.ir import Block, CircuitIR, Component, Net, NetKind, PinAssociation, Provenance, ProvenanceKind

RAILS = frozenset((NetKind.POWER, NetKind.GROUND))
INFERENCE_TOOL = "topology.connectivity"
INFERENCE_VERSION = "1"


def validate_intent(blocks: list[Block], associations: list[PinAssociation], components: list[Component], nets: list[Net]) -> None:
    """Refuse dangling intent with a named reason, without changing its inputs."""
    parts = {c.ref: c for c in components}
    net_names = {n.name for n in nets}
    ids: set[str] = set()
    for b in blocks:
        if not b.id.strip() or b.id in ids:
            raise ValueError(f"intent.block_id: empty or duplicate block id {b.id!r}")
        ids.add(b.id)
        for ref in b.component_refs:
            if ref not in parts:
                raise ValueError(f"intent.component_ref: block {b.id} names unknown component {ref}")
        if len(set(b.component_refs)) != len(b.component_refs):
            raise ValueError(f"intent.component_ref: block {b.id} repeats a member")
        for name in b.input_nets + b.output_nets:
            if name not in net_names:
                raise ValueError(f"intent.net_ref: block {b.id} names unknown net {name}")
    seen: set[tuple[str, str]] = set()
    for a in associations:
        for pin in (a.terminal, a.ic_pin):
            if pin.component_ref not in parts or parts[pin.component_ref].pin(pin.pin_number) is None:
                raise ValueError(f"intent.pin_ref: unknown pin {pin}")
        pair = (str(a.terminal), str(a.ic_pin))
        if pair in seen:
            raise ValueError(f"intent.duplicate_association: {pair[0]} -> {pair[1]}")
        seen.add(pair)
        if a.terminal.component_ref == a.ic_pin.component_ref:
            raise ValueError(f"intent.self_association: {a.terminal} -> {a.ic_pin}")
        if not any(a.terminal in n.pins and a.ic_pin in n.pins for n in nets):
            raise ValueError(f"intent.association_net: {a.terminal} and {a.ic_pin} share no net")


def infer_blocks(components: list[Component], nets: list[Net]) -> list[Block]:
    """Pure non-rail connected components, ordered by block id then natural reference/pin."""
    graph: dict[str, set[str]] = {c.ref: set() for c in components}
    for net in sorted(nets, key=lambda n: n.name):
        if net.kind in RAILS:
            continue
        pins = sorted(net.pins, key=lambda p: (natural_ref_key(p.component_ref), natural_ref_key(p.pin_number)))
        refs = [p.component_ref for p in pins]
        for ref in refs:
            if ref not in graph:
                raise ValueError(f"intent.component_ref: net {net.name} names unknown component {ref}")
            graph[ref].update(refs)
    remaining = set(graph)
    blocks = []
    while remaining:
        seed = min(remaining, key=natural_ref_key)
        members, pending = set(), [seed]
        while pending:
            ref = pending.pop()
            if ref in members:
                continue
            members.add(ref)
            pending.extend(sorted(graph[ref] - members, key=natural_ref_key, reverse=True))
        remaining -= members
        refs = sorted(members, key=natural_ref_key)
        blocks.append(Block(
            id=f"inferred.{seed}", function="connectivity group", component_refs=refs, inferred=True,
            provenance=Provenance(kind=ProvenanceKind.DERIVED, tool=INFERENCE_TOOL, tool_version=INFERENCE_VERSION,
                                  derived_from=refs, note="inferred from connectivity; power and ground excluded; direction unknown",
                                  created_at=datetime(1970, 1, 1, tzinfo=timezone.utc)),
        ))
    return sorted(blocks, key=lambda b: b.id)


@dataclass(frozen=True)
class FunctionalIntent:
    blocks: list[Block]
    pin_associations: list[PinAssociation]
    unassociated: list[str]


def resolve_intent(ir: CircuitIR) -> FunctionalIntent:
    """Use declared blocks or infer them; report rail-only capacitors without a pin assignment.

    Associations are never inferred. Callers opt in explicitly; legacy loading,
    confirmation and compilation never call this function automatically.
    """
    blocks = ir.topology.blocks if ir.topology else []
    associations = ir.topology.pin_associations if ir.topology else []
    validate_intent(blocks, associations, ir.components, ir.nets)
    associated = {a.terminal.component_ref for a in associations}
    unassociated = []
    for c in sorted(ir.components, key=lambda c: natural_ref_key(c.ref)):
        capacitor = "capacitance" in c.electrical or (c.spice is not None and c.spice.device == "C")
        capacitor |= c.symbol is not None and c.symbol.library == "Device" and c.symbol.name in ("C", "C_Polarized", "C_Small")
        connected = [n for n in ir.nets if any(p.component_ref == c.ref for p in n.pins)]
        if capacitor and c.ref not in associated and connected and all(n.kind in RAILS for n in connected):
            unassociated.append(f"{c.ref}: unassociated; shared power/ground rails do not identify the served IC pin")
    return FunctionalIntent(
        blocks=[b.model_copy(deep=True) for b in sorted(blocks, key=lambda b: b.id)] if blocks else infer_blocks(ir.components, ir.nets),
        pin_associations=[a.model_copy(deep=True) for a in sorted(associations, key=lambda a: (
            natural_ref_key(a.terminal.component_ref), natural_ref_key(a.terminal.pin_number),
            natural_ref_key(a.ic_pin.component_ref), natural_ref_key(a.ic_pin.pin_number)))],
        unassociated=unassociated,
    )
