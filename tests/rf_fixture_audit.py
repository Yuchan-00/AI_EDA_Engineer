"""What a top-C fixture leaves out of its circuit: passives on its port nets that are not members, and two networks joined tap to tap.

The kr447 wave-2 review found both on the multiplier chains: the collector
chokes and the next stages' base dividers sat on the tanks' port nets
without being fixture members (the fixture judged a circuit the board does
not have), and the last tank and the band-pass were joined by their tap
capacitors on one net with no resistive node between them (each fixture ran
between its own port models, so their sum was not the cascade). These
helpers state both rules over a built IR, so a block that breaks either
fails its test.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from ai_eda.ir import CircuitIR

#: a two-terminal passive by its reference (R / L / C and digits; a trimmer ``C_T801`` is none of these nets' parts)
PASSIVE_REF = re.compile(r"^[RLC][0-9]+$")


def _net_refs(ir: CircuitIR) -> dict[str, set[str]]:
    return {n.name: {p.component_ref for p in n.pins} for n in ir.nets}


def port_net_outsiders(ir: CircuitIR, network_ids: Iterable[str], *, allowed: Iterable[str] = ()) -> list[tuple[str, str, str]]:
    """``(network, net, ref)`` for every R / L / C with a pin on one of the network's signal ports' nets (kind ``port``) that is not a member.

    ``allowed``: references standing behind a port model on purpose - a
    matched pad whose own fixture proves its input is the port's resistance.
    """
    assert ir.rf is not None
    refs_on = _net_refs(ir)
    skip = set(allowed)
    out: list[tuple[str, str, str]] = []
    for nid in network_ids:
        nw = ir.rf.network(nid)
        assert nw is not None, nid
        members = set(nw.members)
        for port in nw.ports:
            if port.kind != "port":
                continue
            for ref in sorted(refs_on.get(port.net, set())):
                if PASSIVE_REF.match(ref) and ref not in members and ref not in skip:
                    out.append((nid, port.net, ref))
    return out


def nets_joining_networks(ir: CircuitIR, network_ids: Iterable[str]) -> dict[str, list[str]]:
    """Every signal net on which members of two or more of ``network_ids`` meet (two filters joined with no resistive node), with those networks.

    GND and the networks' rail-port nets (the supply every stage's link
    returns to, an ideal source in each fixture) are not signal nets.
    """
    assert ir.rf is not None
    ids = list(network_ids)
    owner: dict[str, str] = {}
    rails = {"GND"}
    for nid in ids:
        nw = ir.rf.network(nid)
        assert nw is not None, nid
        rails |= {p.net for p in nw.ports if p.kind == "rail"}
        for ref in nw.members:
            owner.setdefault(ref, nid)
    out: dict[str, list[str]] = {}
    for net, refs in _net_refs(ir).items():
        if net in rails:
            continue
        hit = sorted({owner[r] for r in refs if r in owner})
        if len(hit) > 1:
            out[net] = hit
    return out
