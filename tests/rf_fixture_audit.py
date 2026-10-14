"""What an RF fixture network leaves out of its circuit, and where two networks meet (the fixture-membership rules of the kr447 boards).

A fixture (``ir.rf.networks``) simulates only its members between its port
models, so a part the board hangs on the network's nets that is not a member
is a part the verdict never saw. The kr447 wave-2 review found both kinds of
defect on the multiplier chains: collector chokes and the next stages' base
dividers on the tanks' port nets without being members (the fixture judged
a circuit the board does not have), and the last tank and the band-pass
joined by their tap capacitors on one net with no resistive node between them
(each fixture ran between its own port models, so their sum was not the
cascade). The fixture-membership pass (part A1) extended the audit to every
network of every build. These helpers state the rules over a built IR; the
test module ``tests/test_rf_fixture_members.py`` applies them to all six
builds with the explicit allowances and the reason each one rests on.

The rules:

1. **Internal nets.** Every net a network's members touch that is not one of
   its ports' nets, GND or a rail / control port net (an ideal DC source in
   the fixture - an ac short) is an internal node: every part on it (test
   points aside: no electrical model) is a member.
2. **Signal port nets.** On the net of a ``port`` (a real resistance: the
   source or load the fixture puts there stands for the part behind it) or
   of a ``probe`` (no load: it stands for a high-impedance reader) every
   two-terminal passive - R, L, C, a trimmer C_T - is a member, unless the
   port model is stated to stand for it too. Transistors, ICs and connectors
   there are what the port model stands for.
3. **Junctions.** Two networks meet on a signal net (each has a member there
   the other lacks) only when their members are nested (a sub-network fixture:
   the enclosing network judges the cascade), when a third network holds
   both sides' members there (a cascade fixture judges the junction), or at a
   junction the caller allows with its reason.

``allowed`` / reasons are the caller's (the test module): a helper here only
finds; it never decides that something is fine.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from ai_eda.ir import CircuitIR
from ai_eda.ir.rf import RFNetwork

#: a two-terminal passive by its reference: R / L / C and digits, or a trimmer C_T and digits
PASSIVE_REF = re.compile(r"^(?:[RLC]|C_T)[0-9]+$")
#: a test point: no electrical model, never a member
TEST_POINT_REF = re.compile(r"^TP[0-9]+$")
GROUND = "GND"


@dataclass(frozen=True)
class Outsider:
    """A part on one of a network's signal nets that is not a member: ``port`` names the network's port there (``None``: an internal net)."""

    network: str
    net: str
    ref: str
    port: str | None


@dataclass(frozen=True)
class Junction:
    """A signal net on which two networks meet (each has a member there the other lacks)."""

    net: str
    networks: tuple[str, str]
    #: the two sides' members on the net
    sides: tuple[tuple[str, ...], tuple[str, ...]]
    #: one network's members contain the other's (a sub-network fixture)
    nested: bool
    #: the other networks that hold both sides' members on the net (cascade fixtures)
    covered_by: tuple[str, ...]


def net_refs(ir: CircuitIR) -> dict[str, set[str]]:
    """Net name -> the references with a pin on it."""
    return {n.name: {p.component_ref for p in n.pins} for n in ir.nets}


def _networks(ir: CircuitIR, network_ids: Iterable[str] | None) -> list[RFNetwork]:
    assert ir.rf is not None
    if network_ids is None:
        return list(ir.rf.networks)
    out = []
    for nid in network_ids:
        nw = ir.rf.network(nid)
        assert nw is not None, nid
        out.append(nw)
    return out


def dc_nets(nw: RFNetwork) -> set[str]:
    """GND and the nets of the network's rail / control ports (ideal DC sources in the fixture: ac shorts)."""
    return {GROUND} | {p.net for p in nw.ports if p.kind in ("rail", "control")}


def signal_nets(ir: CircuitIR, nw: RFNetwork) -> dict[str, str | None]:
    """Every net the network's members touch, less its DC nets -> the name of the ``port`` / ``probe`` on it (``None``: an internal net)."""
    members = set(nw.members)
    dc = dc_nets(nw)
    ports = {p.net: p.name for p in nw.ports if p.kind in ("port", "probe")}
    out: dict[str, str | None] = {}
    for n in ir.nets:
        if n.name in dc or not any(p.component_ref in members for p in n.pins):
            continue
        out[n.name] = ports.get(n.name)
    # a port on a net no member touches is still one of the network's signal nets
    for net, name in ports.items():
        out.setdefault(net, name)
    return out


def outsiders(ir: CircuitIR, network_ids: Iterable[str] | None = None) -> list[Outsider]:
    """Rules 1 and 2 (module docstring): every part on a network's internal nets and every passive on its port nets that is not a member."""
    refs_on = net_refs(ir)
    out: list[Outsider] = []
    for nw in _networks(ir, network_ids):
        members = set(nw.members)
        for net, port in sorted(signal_nets(ir, nw).items()):
            for ref in sorted(refs_on.get(net, set()) - members):
                if TEST_POINT_REF.match(ref):
                    continue
                if port is not None and not PASSIVE_REF.match(ref):
                    continue  # a transistor, IC or connector on a port net is what the port model stands for
                out.append(Outsider(nw.id, net, ref, port))
    return out


def junctions(ir: CircuitIR, network_ids: Iterable[str] | None = None) -> list[Junction]:
    """Rule 3 (module docstring): every signal net shared by two networks where each has a member the other lacks, with nesting and cover."""
    nws = _networks(ir, network_ids)
    everything = list(ir.rf.networks) if ir.rf is not None else []
    refs_on = net_refs(ir)
    sig = {nw.id: signal_nets(ir, nw) for nw in nws}
    out: list[Junction] = []
    for i, a in enumerate(nws):
        for b in nws[i + 1:]:
            ma, mb = set(a.members), set(b.members)
            for net in sorted(set(sig[a.id]) & set(sig[b.id])):
                on = refs_on.get(net, set())
                side_a, side_b = (on & ma) - mb, (on & mb) - ma
                if not side_a or not side_b:
                    continue
                need = (on & ma) | (on & mb)
                cover = tuple(sorted(c.id for c in everything if c.id not in (a.id, b.id) and need <= set(c.members)))
                out.append(Junction(net, (a.id, b.id), (tuple(sorted(side_a)), tuple(sorted(side_b))), ma <= mb or mb <= ma, cover))
    return out


# --------------------------------------------------------------------------- the wave-2 helpers (kept: the template tests use them)


def port_net_outsiders(ir: CircuitIR, network_ids: Iterable[str], *, allowed: Iterable[str] = ()) -> list[tuple[str, str, str]]:
    """``(network, net, ref)`` for every passive on one of the networks' ``port`` nets that is not a member (rule 2), less ``allowed`` references."""
    skip = set(allowed)
    return [(o.network, o.net, o.ref) for o in outsiders(ir, network_ids) if o.port is not None and o.ref not in skip]


def nets_joining_networks(ir: CircuitIR, network_ids: Iterable[str]) -> dict[str, list[str]]:
    """Every signal net on which two or more of ``network_ids`` meet with no network holding both sides (rule 3 without nesting or cover)."""
    out: dict[str, list[str]] = {}
    for j in junctions(ir, network_ids):
        if j.nested or j.covered_by:
            continue
        have = out.setdefault(j.net, [])
        have += [n for n in j.networks if n not in have]
    return {net: sorted(ids) for net, ids in out.items()}
