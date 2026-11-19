"""Blocks: which glyphs are drawn together, in which order, and where a block meets the others.

:func:`partition` returns the blocks in drawing order. Membership: the declared ``topology.blocks`` first, in declaration
order (a ref belongs to the first block that names it; unknown refs are dropped with a note; a block id declared again
is merged into the first, noted), then
:func:`~ai_eda.tools.functional_intent.infer_blocks` over the parts no block names (noted "inferred"). Test points
(:func:`is_test_point`: a one-pin ``TestPoint`` symbol of a ``Connector*`` library) are taken out of every block, declared
or not, and drawn together in one block :data:`TEST_POINT_BLOCK`, last, each named as the sheet names its net (the recipe
:class:`~ai_eda.compilers.textbook.recipes.TestPointRecipe`): a pad on a net is no stage of the circuit, and inside a
stage's block it breaks the stage's recipe (a supply chain with a part off its line). Units of a
multi-unit part move: an amplifier unit to the block with the most parts on its signal nets (a two-terminal part across
its output and one of its inputs, its feedback, goes with it), or, when its loop closes through followers (see
:func:`follower_loop`), to the followers' block with the parts that join their node to its input; the power unit with the lowest used amplifier unit (all
noted); a signal net a moved unit now shares with the block it left becomes an input of
the one block and an output of the other, by the unit's pin type, so the block order keeps the signal flow. The block DAG has an edge A -> B when a non-ground net leaves A as an output and
enters B as an input (declared nets, else output / power-output pins feeding input / power-input pins); a block made
only of connectors that holds another block's input is a source. Edges that would close a cycle are dropped, forward
edges (declaration order) first, with a note. ``layer`` is the longest path; the order is (layer, declaration index). An
inferred block of connectors only takes its nets as outputs (a source) or inputs (a sink), so its ports face its block.

:func:`ports` gives where a drawing meets the rest of the sheet (inputs left, outputs right, rails on top, other nets
leaving the block at the bottom); :func:`satellites` finds blocks hung under (or over) a host's bottom (top) port.
"""

from __future__ import annotations

from dataclasses import replace

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.compilers.textbook.glyphs import pin_cell
from ai_eda.compilers.textbook.model import BlockInfo, Drawing, Kind, Model, Port, Role
from ai_eda.compilers.textbook.model import is_test_point as model_is_test_point
from ai_eda.tools.functional_intent import infer_blocks

#: the block every test point is drawn in (module docstring)
TEST_POINT_BLOCK = "test_points"

DRIVERS = ("output", "power_out")
DRIVEN = ("input", "power_in")


def is_test_point(model: Model, key: str) -> bool:
    """:func:`~ai_eda.compilers.textbook.model.is_test_point` of glyph ``key``."""
    return model_is_test_point(model.glyphs[key])


def partition(model: Model) -> tuple[BlockInfo, ...]:
    blocks = _assign_units(model, _membership(model))
    return _layered(model, blocks)


def _membership(model: Model) -> list[BlockInfo]:
    ir = model.ir
    keys_of: dict[str, list[str]] = {}
    for g in model.glyphs.values():
        keys_of.setdefault(g.ref, []).append(g.key)
    declared = list(ir.topology.blocks) if ir is not None and ir.topology else []
    board_only = {c.ref for c in ir.components if c.board_only} if ir is not None else set()
    claimed: set[str] = set()
    out: list[BlockInfo] = []
    test_points = sorted((k for k in model.glyphs if is_test_point(model, k)), key=natural_ref_key)
    claimed |= {model.glyphs[k].ref for k in test_points}
    for index, block in enumerate(declared):
        members: list[str] = []
        unknown = []
        for ref in sorted(set(block.component_refs), key=natural_ref_key):
            if ref not in keys_of:
                if ref not in board_only and ref not in model.aliases:  # an alias is drawn by its combination's outline
                    unknown.append(ref)
                continue
            if ref not in claimed:
                claimed.add(ref)
                members += keys_of[ref]
        if unknown:
            model.notes.append(f"block {block.id}: unknown part(s) {', '.join(unknown)} dropped")
        first = next((i for i, b in enumerate(out) if b.id == block.id), None)
        if first is not None:  # one id, one block: the parts of a repeated declaration join the first
            b = out[first]
            out[first] = replace(b, members=b.members + tuple(members), inputs=tuple(dict.fromkeys(b.inputs + tuple(block.input_nets))),
                                 outputs=tuple(dict.fromkeys(b.outputs + tuple(block.output_nets))))
            model.notes.append(f"block {block.id}: declared more than once; its parts are drawn in one block")
            continue
        if members:
            out.append(BlockInfo(block.id, tuple(members), tuple(block.input_nets), tuple(block.output_nets), declared=True, layer=index))
    rest = [g.component for g in model.glyphs.values() if g.ref not in claimed and g.key == keys_of[g.ref][0]]
    if rest:
        refs = {c.ref for c in rest}
        nets = [n.model_copy(update={"pins": [p for p in n.pins if p.component_ref in refs]}) for n in ir.nets]
        inferred = infer_blocks(rest, [n for n in nets if n.pins])
        for i, block in enumerate(inferred):
            members = [k for ref in block.component_refs for k in keys_of[ref]]
            out.append(BlockInfo(block.id, tuple(members), inferred=True, layer=len(declared) + i))
        model.notes.append(f"blocks inferred from connectivity for {len(rest)} part(s) no declared block names: "
                           f"{', '.join(b.id for b in inferred)}")
    if test_points:
        out.append(BlockInfo(TEST_POINT_BLOCK, tuple(test_points), layer=len(out) + len(declared)))
        model.notes.append(f"block {TEST_POINT_BLOCK}: test point(s) {', '.join(model.glyphs[k].ref for k in test_points)} drawn together, "
                           "each on a label or power symbol of its net (taken out of the blocks that named them)")
    return out


def _signal_nets(model: Model, key: str) -> set[str]:
    return {n for n in model.glyphs[key].net.values() if model.nets[n].role is Role.SIGNAL}


def _assign_units(model: Model, blocks: list[BlockInfo]) -> list[BlockInfo]:
    """Amplifier units go to the block with the most parts on their signal nets (ties: where they are), their feedback
    parts with them; a power unit goes with the lowest used amplifier unit of its part."""
    block_of = {k: b.id for b in blocks for k in b.members}
    units: dict[str, list[str]] = {}
    for g in model.glyphs.values():
        units.setdefault(g.ref, []).append(g.key)
    order = [b.id for b in blocks]
    moved: list[tuple[str, str, str]] = []
    for ref in sorted(units, key=natural_ref_key):
        keys = sorted(units[ref], key=lambda k: model.glyphs[k].unit)
        if len(keys) < 2:
            continue
        for key in keys:
            if model.glyphs[key].kind is not Kind.AMP:
                continue
            count: dict[str, set[str]] = {}
            for net in _signal_nets(model, key):
                for other, _ in model.nets[net].pins:
                    if model.glyphs[other].ref != ref:
                        count.setdefault(block_of[other], set()).add(model.glyphs[other].ref)
            if not count:
                continue
            here = block_of[key]
            best = max(order, key=lambda b: (len(count.get(b, ())), b == here, -order.index(b)))
            loop = follower_loop(model, key)
            why = "most parts on its signal nets"
            if loop is not None and len({block_of[k] for k in loop[0]}) == 1:  # drawn with the followers its loop runs through
                if block_of[loop[0][0]] != best or len(count.get(best, ())) <= len(count.get(here, ())):
                    best, why = block_of[loop[0][0]], f"its loop runs through {'/'.join(loop[0])}"
            else:
                loop = None
            if best != here and (len(count.get(best, ())) > len(count.get(here, ())) or loop is not None):
                moved.append((key, here, best))
                block_of[key] = best
                model.notes.append(f"unit {model.glyphs[key].shown_ref} drawn in block {best} ({why})")
                for part in _feedback(model, key) + (loop[1] if loop is not None else []):
                    if block_of[part] == here:  # the stage's feedback stays with its amplifier
                        moved.append((part, here, best))
                        block_of[part] = best
                        model.notes.append(f"{part} drawn in block {best} with {model.glyphs[key].shown_ref} (its feedback)")
        used = [k for k in keys if model.glyphs[k].kind is Kind.AMP and _signal_nets(model, k)]
        for key in keys:
            if model.glyphs[key].kind is Kind.PWRUNIT and used and block_of[key] != block_of[used[0]]:
                block_of[key] = block_of[used[0]]
                model.notes.append(f"unit {model.glyphs[key].shown_ref} drawn in block {block_of[key]} with {model.glyphs[used[0]].shown_ref}")
    ins: dict[str, list[str]] = {b.id: list(b.inputs) for b in blocks}
    outs: dict[str, list[str]] = {b.id: list(b.outputs) for b in blocks}
    for key, was, now in moved:
        left = {n for k, b in block_of.items() if b == was for n in model.glyphs[k].net.values()}
        for number, net in sorted(model.glyphs[key].net.items()):
            if model.nets[net].role is not Role.SIGNAL or net not in left:
                continue
            etype = model.glyphs[key].pin(number).etype
            src, dst = (now, was) if etype in DRIVERS else (was, now) if etype in DRIVEN else (None, None)
            if src is None or net in outs[src] and net in ins[dst]:
                continue
            outs[src].append(net) if net not in outs[src] else None
            ins[dst].append(net) if net not in ins[dst] else None
            model.notes.append(f"block order: {net} runs from block {src} to block {dst} after unit {model.glyphs[key].shown_ref} moved")
    out = []
    for b in blocks:
        members = tuple(sorted((k for k in block_of if block_of[k] == b.id), key=natural_ref_key))
        if members:
            out.append(replace(b, members=members, inputs=tuple(ins[b.id]), outputs=tuple(outs[b.id])))
    return out


def follower_loop(model: Model, key: str) -> tuple[list[str], list[str]] | None:
    """(followers, feedback parts) when amplifier unit ``key``'s loop closes through followers: transistors whose control
    pins sit on the unit's output and whose through pairs run from one node to a rail each, the node an input of the unit
    or joined to one by two-terminal parts (the feedback parts). None otherwise."""
    g = model.glyphs[key]
    io = g.amp_io()
    if io is None or any(p.number not in g.net for p in io):
        return None
    out = g.net[io[0].number]
    rail = lambda n: model.nets[n].role is not Role.SIGNAL  # noqa: E731

    def followers(node: str) -> list[str] | None:
        found = []
        for k in sorted({k for k, _ in model.nets[node].pins if k != key}, key=natural_ref_key):
            q = model.glyphs[k]
            if q.kind is Kind.TWO:
                continue
            pair, ctrl = q.through(), q.control()
            if q.kind is not Kind.THRU3 or pair is None or ctrl is None or q.net.get(ctrl.number) != out:
                return None
            ends = [q.net.get(p.number) for p in pair]
            if None in ends or node not in ends or not all(n == node or rail(n) for n in ends):
                return None
            found.append(k)
        return found or None

    inputs = sorted(io[1:], key=lambda p: p.name.strip() not in ("-", "~"))  # the inverting input first
    for p in inputs:
        node = g.net[p.number]
        if node == out or rail(node):
            continue
        direct = followers(node)
        if direct:
            return direct, []
        far: dict[str, list[str]] = {}
        for k, _ in model.nets[node].pins:
            q = model.glyphs[k]
            nets = set(q.net.values())
            if k != key and q.kind is Kind.TWO and len(q.net) == 2 and len(nets) == 2:
                n = next(iter(nets - {node}))
                if n != out and not rail(n):
                    far.setdefault(n, []).append(k)
        for n in sorted(far):
            found = followers(n)
            if found:
                return found, sorted(set(far[n]), key=natural_ref_key)
    return None


def _feedback(model: Model, key: str) -> list[str]:
    """The two-terminal parts joining amplifier unit ``key``'s output to one of its inputs."""
    g = model.glyphs[key]
    io = g.amp_io()
    if io is None or io[0].number not in g.net:
        return []
    out = g.net[io[0].number]
    ins = {g.net[p.number] for p in io[1:] if p.number in g.net}
    return sorted((k for k, o in model.glyphs.items() if o.kind is Kind.TWO and len(set(o.net.values())) == 2
                   and out in o.net.values() and set(o.net.values()) - {out} <= ins), key=natural_ref_key)


def _nets_of(model: Model, block: BlockInfo) -> set[str]:
    """The nets a block's glyphs are on; a test point's net is not one (it is named by the pad's label, no link of the block)."""
    return {n for k in block.members if not is_test_point(model, k) for n in model.glyphs[k].net.values()}


def _pin_types(model: Model, block: BlockInfo, net: str) -> set[str]:
    members = set(block.members)
    return {model.glyphs[k].pin(p).etype for k, p in model.nets[net].pins if k in members}


def connectors_only(model: Model, block: BlockInfo) -> bool:
    return bool(block.members) and all(model.glyphs[k].kind is Kind.CONN for k in block.members)


def _source(model: Model, block: BlockInfo, blocks: list[BlockInfo]) -> bool:
    """A block made only of connectors that holds another block's input."""
    if not connectors_only(model, block):
        return False
    nets = _nets_of(model, block)
    return any(nets & set(other.inputs) for other in blocks if other.id != block.id)


def edges(model: Model, blocks: list[BlockInfo]) -> list[tuple[str, str]]:
    """Block DAG edges before cycle removal, in a fixed order. A net leaves a block as a declared output or from an output
    pin; it enters a block as a declared input, at an input pin, or at a connector of a block made only of connectors."""
    nets = {b.id: _nets_of(model, b) for b in blocks}
    sources = {b.id for b in blocks if _source(model, b, blocks)}
    sinks = {b.id for b in blocks if connectors_only(model, b) and b.id not in sources}
    out = []
    for a in blocks:
        for b in blocks:
            if a.id == b.id or b.id in sources or a.id in sinks:
                continue
            shared = sorted(n for n in nets[a.id] & nets[b.id] if model.nets[n].role is not Role.GROUND)
            for n in shared:
                leaves = n in a.outputs or (not a.declared and bool(_pin_types(model, a, n) & set(DRIVERS))) or a.id in sources
                enters = n in b.inputs or (not b.declared and bool(_pin_types(model, b, n) & set(DRIVEN))) or b.id in sinks
                if leaves and enters:
                    out.append((a.id, b.id))
                    break
    return out


def _reaches(succ: dict[str, set[str]], a: str, b: str) -> bool:
    seen, stack = {a}, [a]
    while stack:
        x = stack.pop()
        if x == b:
            return True
        for y in sorted(succ.get(x, ())):
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return False


def dag(model: Model, blocks: list[BlockInfo]) -> dict[str, set[str]]:
    """The block DAG: forward edges (declaration order) first, then backward ones that close no cycle; every dropped edge
    is noted."""
    index = {b.id: i for i, b in enumerate(blocks)}
    succ: dict[str, set[str]] = {b.id: set() for b in blocks}
    found = edges(model, blocks)
    for a, b in sorted(found, key=lambda e: (index[e[0]] > index[e[1]], index[e[0]], index[e[1]])):
        if _reaches(succ, b, a):
            model.notes.append(f"block order: edge {a} -> {b} dropped (it would close a cycle)")
            continue
        succ[a].add(b)
    return succ


def layers(blocks: list[BlockInfo], succ: dict[str, set[str]]) -> dict[str, int]:
    """Longest path from the sources (sorted Kahn)."""
    index = {b.id: i for i, b in enumerate(blocks)}
    indeg = {b.id: 0 for b in blocks}
    for a in succ:
        for b in succ[a]:
            indeg[b] += 1
    layer = {b.id: 0 for b in blocks}
    ready = sorted((b for b in indeg if not indeg[b]), key=index.get)
    while ready:
        a = ready.pop(0)
        for b in sorted(succ[a], key=index.get):
            layer[b] = max(layer[b], layer[a] + 1)
            indeg[b] -= 1
            if not indeg[b]:
                ready.append(b)
                ready.sort(key=index.get)
    return layer


def _layered(model: Model, blocks: list[BlockInfo]) -> tuple[BlockInfo, ...]:
    succ = dag(model, blocks)
    layer = layers(blocks, succ)
    index = {b.id: i for i, b in enumerate(blocks)}
    # the test points' block last whatever its layer (module text): it links to no block, and the drawings before it keep
    # their order and their power symbols' numbers
    ordered = sorted(blocks, key=lambda b: (b.id == TEST_POINT_BLOCK, layer[b.id], index[b.id]))
    out = []
    for b in ordered:
        if not b.declared and connectors_only(model, b):  # a source's nets leave it, a sink's enter it
            nets = tuple(sorted(n for n in _nets_of(model, b) if model.nets[n].role is not Role.GROUND))
            b = replace(b, outputs=nets) if _source(model, b, blocks) else replace(b, inputs=nets)
        out.append(replace(b, layer=layer[b.id]))
    return tuple(out)


def ports(model: Model, block: BlockInfo, drawing: Drawing) -> tuple[Port, ...]:
    """Where ``drawing`` (block-relative) meets the rest of the sheet: each net of its glyphs that has a pin outside the
    drawing, at its first pin's height - a block input on the left edge, an output on the right edge; a rail on top and any
    other net at the bottom, both at that pin."""
    if drawing.box is None:
        return ()
    x0, _, x1, _ = drawing.box
    inside = set(drawing.place)
    out = []
    for net in sorted({n for k in inside for n in model.glyphs[k].net.values()}):
        info = model.nets[net]
        if info.role is Role.GROUND or all(k in inside for k, _ in info.pins) or not any(k in inside for k, _ in info.pins):
            continue  # (no pin of the net in the drawing: a test point's net, named by its label - model.is_test_point)
        key, number = next((k, p) for k, p in info.pins if k in inside and (model.glyphs[k].ref, p) not in model.stacked)
        g, place = model.glyphs[key], drawing.place[key]
        at = pin_cell(g, number, (place.x, place.y), place.orient)
        if info.role in (Role.SUPPLY, Role.NEGATIVE):
            out.append(Port(net, "T", at, "rail"))
        elif net in block.inputs:
            out.append(Port(net, "L", (x0, at[1]), "in"))
        elif net in block.outputs:
            out.append(Port(net, "R", (x1, at[1]), "out"))
        else:
            out.append(Port(net, "B", at, "slot"))
    return tuple(out)


def satellites(model: Model, blocks: tuple[BlockInfo, ...], ports_of: dict[str, tuple[Port, ...]]) -> dict[str, tuple[str, Port]]:
    """Block id -> (host id, host port) for each block whose only signal link to the others is one net that is a bottom
    (or top) port of exactly one other block."""
    nets = {b.id: {n for n in _nets_of(model, b) if model.nets[n].role is Role.SIGNAL} for b in blocks}
    plain = {b.id for b in blocks if not connectors_only(model, b)}
    out: dict[str, tuple[str, Port]] = {}
    for b in blocks:
        links = {n for n in nets[b.id] if any(n in nets[o.id] for o in blocks if o.id != b.id)}
        if len(links) != 1 or b.id not in plain:
            continue
        net = next(iter(links))
        hosts = [(o.id, p) for o in blocks if o.id != b.id and o.id not in out and o.id in plain for p in ports_of.get(o.id, ()) if p.net == net]
        if len(hosts) == 1 and hosts[0][1].side in ("B", "T") and hosts[0][0] not in out:
            out[b.id] = hosts[0]
    return out
