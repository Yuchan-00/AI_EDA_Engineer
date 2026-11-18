"""Opt-in rail/column schematic policy, version 0.9.

CHOICE: at most 20 physical parts and one IC (a non-connector with more than
three pins). Library geometry and connectivity, never template names, select
the motifs. All positions and route vertices use the 1.27 mm grid. Larger
designs retain compiler 0.8. No electrical parts are added to the design.
"""

from __future__ import annotations

from collections import defaultdict, deque
from itertools import combinations
from typing import TYPE_CHECKING

from ai_eda.compilers.ids import net_item_uuid, sheet_uuid
from ai_eda.compilers.pins import load_verified_symbol, pad_pin_types
from ai_eda.compilers.schematic_layout import GRID_MM, Extent, natural_ref_key, pin_position, snap
from ai_eda.errors import CompileError
from ai_eda.gui import schematic_render as render
from ai_eda.ir import CircuitIR, Component, LibraryRef, NetKind, Pin, PinRef, Provenance, ProvenanceKind
from ai_eda.tools.kicad import sexpr
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.kicad.sexpr import Node, Q, S

if TYPE_CHECKING:
    from ai_eda.compilers.schematic import SchematicCompiler

PinKey = tuple[str, str]
Point = tuple[int, int]

VERSION = "0.9"
MAX_PARTS = 20


def connector(c: Component) -> bool:
    return c.symbol is not None and c.symbol.library.startswith("Connector")


def eligible(ir: CircuitIR) -> bool:
    return len(ir.components) <= MAX_PARTS and sum(len(c.pins) > 3 and not connector(c) for c in ir.components) <= 1


def mm(x: float) -> float:
    return snap(x * GRID_MM)


def cell(p: tuple[float, float]) -> Point:
    return tuple(round(v / GRID_MM) for v in p)


def on(a: Point, b: Point, p: Point) -> bool:
    return (b[0] - a[0]) * (p[1] - a[1]) == (b[1] - a[1]) * (p[0] - a[0]) and all(min(a[i], b[i]) <= p[i] <= max(a[i], b[i]) for i in (0, 1))


class Drawing:
    def __init__(self, compiler: SchematicCompiler, ir: CircuitIR, library: KicadLibrary) -> None:
        from ai_eda.compilers.schematic import _PlacedSymbol
        self.compiler, self.ir, self.library = compiler, ir, library
        self.owner = compiler._pin_net_names(ir)
        self.placed = {}
        self.segments = defaultdict(list)
        self.notes = []
        self.field_side = {}
        self.ground_pairs = []
        self.port_specs = []
        for c in compiler._checked_components(ir):
            symbol = load_verified_symbol(c, library)
            pad_pin_types(c, symbol)
            self.placed[c.ref] = _PlacedSymbol(c, symbol, compiler._footprint_id(c, library), 0, 0, Extent(0, 0, 0, 0))
        # Validate every net, missing pin and no-connect before attempting geometry.
        pins = compiler._pin_map(self.placed)
        compiler._connectivity(ir, self.placed, pins, compiler._stacks(self.placed, pins, self.owner))
        self.grounds = {n.name for n in ir.nets if n.kind == NetKind.GROUND}
        self.supplies = {n.name for n in ir.nets if n.kind == NetKind.POWER}
        self.connectors = [r for r, ps in self.placed.items() if connector(ps.component)]
        self.parts = [r for r in self.placed if r not in self.connectors]

    def key(self, ref: str, name: str) -> PinKey:
        pins = [p.number for p in self.placed[ref].symbol.pins if p.name == name]
        if len(pins) != 1:
            raise CompileError(f"textbook: {ref} has no unique {name} pin")
        return ref, pins[0]

    def nets(self, ref: str) -> set[str]:
        return {self.owner[k] for k in self.owner if k[0] == ref}

    def pin(self, key: PinKey) -> Point:
        ps = self.placed[key[0]]
        p = next(p for p in ps.symbol.pins if p.number == key[1])
        actual = pin_position(ps.x, ps.y, ps.rotation, ps.mirror, p)
        point = cell(actual)
        if any(abs(mm(v) - coordinate) > 1e-6 for v, coordinate in zip(point, actual)):
            raise CompileError(f"textbook: library pin {key} at {actual} is off the 1.27 mm grid")
        return point

    def put(self, ref: str, x: float, y: float, rotation: int = 0, mirror: str | None = None) -> None:
        ps = self.placed[ref]
        ps.x, ps.y, ps.rotation, ps.mirror = mm(x), mm(y), rotation, mirror

    def pair(self, ref: str, first_net: str, x: float, y: float, vertical: bool) -> None:
        """Orient a real two-terminal symbol with first_net above or left."""
        keys = [(ref, p.number) for p in self.placed[ref].symbol.pins]
        first = next(k for k in keys if self.owner.get(k) == first_net)
        other = next(k for k in keys if k != first)
        for rot in (0, 90, 180, 270):
            self.put(ref, x, y, rot)
            a, b = self.pin(first), self.pin(other)
            if (a[0] == b[0] and a[1] < b[1]) if vertical else (a[1] == b[1] and a[0] < b[0]):
                return
        raise CompileError(f"textbook: {ref} does not have opposed two-terminal pins")

    def line(self, net: str, *points: Point) -> None:
        for a, b in zip(points, points[1:]):
            if a != b:
                self.segments[net].append((a, b))

    def astable(self) -> bool:
        transistors = [r for r in self.parts if {p.name for p in self.placed[r].symbol.pins} == {"B", "C", "E"}]
        if len(transistors) != 2 or len(self.parts) not in (8, 10) or len(self.supplies) != 1:
            return False
        supply = next(iter(self.supplies))
        q1, q2 = transistors
        pairs = []
        tails = []
        for q, other in ((q1, q2), (q2, q1)):
            if any(self.key(ref, name) not in self.owner for ref in (q, other) for name in ("B", "C", "E")):
                return False
            cn, bn = self.owner[self.key(q, "C")], self.owner[self.key(other, "B")]
            find = lambda nets: [r for r in self.parts if len(self.placed[r].symbol.pins) == 2 and self.nets(r) == nets]
            emitter = self.owner[self.key(q, "E")]
            tail = [r for g in sorted(self.grounds) for r in find({emitter, g})] if emitter not in self.grounds else []
            if emitter not in self.grounds and len(tail) != 1:
                return False
            tails.append(tail)
            load, bias, coupling = find({supply, cn}), find({supply, bn}), find({cn, bn})
            if not all(len(v) == 1 for v in (load, bias, coupling)):
                return False
            pairs.append((q, load[0], bias[0], coupling[0], cn, bn))
        members = {r for pair in pairs for r in pair[:4]} | {r for tail in tails for r in tail}
        if members != set(self.parts) or len(tails[0]) != len(tails[1]):
            return False
        # Verify the core graph automorphism including symbol and value colours.
        for a, b in zip([*pairs[0][:4], *tails[0]], [*pairs[1][:4], *tails[1]]):
            pa, pb = self.placed[a], self.placed[b]
            if (pa.symbol.lib_id, pa.component.value) != (pb.symbol.lib_id, pb.component.value):
                return False
        self.put(q1, 42, 86, mirror="y")
        self.put(q2, 114, 86)
        for i, (q, load, bias, coupling, cn, bn) in enumerate(pairs):
            outer, inner = (40, 66) if i == 0 else (116, 90)
            self.put(q, outer + (2 if i == 0 else -2), 86, mirror="y" if i == 0 else None)
            self.field_side[q] = -1 if i == 0 else 1
            self.pair(load, supply, outer, 43, True)
            self.pair(bias, supply, inner, 43, True)
            if tails[i]:
                self.pair(tails[i][0], self.owner[self.key(q, "E")], outer, 104, True)
            self.pair(coupling, cn if i == 0 else bn, (outer + inner) // 2, 62, False)
            collector = self.pin(self.key(q, "C"))
            self.line(cn, collector, (outer, 62))
            for r in (load, coupling):
                k = next(k for k in self.owner if k[0] == r and self.owner[k] == cn)
                self.line(cn, self.pin(k), (outer, 62))
            for r in (bias, coupling):
                k = next(k for k in self.owner if k[0] == r and self.owner[k] == bn)
                self.line(bn, self.pin(k), (inner, 62))
            other = q2 if i == 0 else q1
            base = self.pin(self.key(other, "B"))
            self.line(bn, (inner, 62), (inner, 68), (90 if i == 0 else 66, 86), base)
        self.notes.append("mirrored core: verified symbol/value-coloured column swap; one cross-coupling crossing")
        return True

    def place(self) -> None:
        if self.astable():
            self.place_connectors(142, 60)
            return
        if self.integrated_circuit():
            return
        bjts = [r for r in self.parts if {p.name for p in self.placed[r].symbol.pins} == {"B", "C", "E"}
                and all((r, p.number) in self.owner for p in self.placed[r].symbol.pins)]
        if len(bjts) == 1:
            q = bjts[0]
            self.put(q, 78, 72)
            bn, cn, en = (self.owner[self.key(q, name)] for name in ("B", "C", "E"))
            for i, r in enumerate(r for r in self.parts if r != q):
                nets = self.nets(r)
                if bn in nets and len(nets) == 2:
                    self.pair(r, next(n for n in nets if n != bn), 52, 72, False)
                elif cn in nets and nets & self.supplies:
                    self.pair(r, min(nets & self.supplies), 80, 46, True)
                elif en not in self.grounds and en in nets and nets & self.grounds and len(self.placed[r].symbol.pins) == 2:
                    self.pair(r, en, 80, 98, True)
                else:
                    self.put(r, 110 + 28 * i, 60)
            for i, r in enumerate(self.connectors):
                self.put(r, 24 if i == 0 else 128, 72 if i == 0 else 44, mirror="y" if i == 0 else None)
                if i == 0:
                    self.field_side[r] = -1
            return
        # Passive chains: shortest net distance from the first declared input,
        # else connector pin order. Rails do not participate in signal ranking.
        inputs = [n for b in (self.ir.topology.blocks if self.ir.topology else []) for n in b.input_nets if n not in self.grounds]
        ck = sorted((k for k in self.owner if k[0] in self.connectors), key=lambda k: (natural_ref_key(k[0]), natural_ref_key(k[1])))
        start = inputs[0] if inputs else next((self.owner[k] for k in ck if self.owner[k] not in self.grounds), next(iter(sorted(self.owner.values())), None))
        if start is None:
            for i, r in enumerate(self.placed):
                self.put(r, 40 + 30 * i, 50)
            return
        distance = {start: 0}
        queue = deque([start])
        while queue:
            net = queue.popleft()
            for r in self.parts:
                if net in self.nets(r):
                    for n in sorted(self.nets(r) - self.grounds):
                        if n not in distance:
                            distance[n] = distance[net] + 1
                            queue.append(n)
        vertical = bool(self.supplies)
        lanes = defaultdict(int)
        net_column = {}
        part_order = sorted(self.parts, key=natural_ref_key, reverse=vertical)
        for r in sorted(part_order, key=lambda r: min((distance.get(n, 99) for n in self.nets(r)), default=99)):
            nets = self.nets(r)
            if not nets:
                self.put(r, 40 + 30 * self.parts.index(r), 110)
                continue
            first = min(nets, key=lambda n: (n in self.grounds, distance.get(n, 99), n))
            rank = distance.get(first, 0)
            lane = lanes[rank]
            lanes[rank] += 1
            if len(self.placed[r].symbol.pins) == 2:
                if vertical:
                    x = net_column.get(first, 60 + 30 * lane)
                    self.pair(r, first, x, 44 + 26 * rank, True)
                    for n in sorted(nets - self.supplies - self.grounds):
                        net_column.setdefault(n, x)
                elif nets & self.grounds:
                    self.pair(r, first, 46 + 30 * rank + 22 * lane, 70, True)
                else:
                    self.pair(r, first, 46 + 30 * rank, 50, False)
            else:
                self.put(r, 76 + 40 * lane, 60 + 36 * rank)
        right = max(cell((p.x, p.y))[0] for p in self.placed.values()) + 36
        if len(self.connectors) == 1:
            self.place_connectors(right, 50)
        else:
            for i, r in enumerate(self.connectors):
                self.put(r, 22 if i == 0 else right, 50 + 20 * max(0, i - 1), mirror="y" if i == 0 else None)
                if i == 0:
                    self.field_side[r] = -1
        if not vertical:
            # The signal backbone stays horizontal even when an edge header
            # carries both input and output. Its input gets an edge terminal.
            if len(self.connectors) <= 1:
                self.port_specs.append((start, (24, 50)))
            for r in self.parts:
                ps = self.placed[r]
                if len(ps.symbol.pins) != 2:
                    continue
                for p in ps.symbol.pins:
                    key = r, p.number
                    if key in self.owner and self.owner[key] not in self.grounds:
                        at = self.pin(key)
                        self.line(self.owner[key], at, (at[0], 50))

    def integrated_circuit(self) -> bool:
        cores = [r for r in self.parts if len(self.placed[r].symbol.pins) > 3]
        if not cores:
            return False
        core = cores[0]
        self.put(core, 100, 80)
        symbol = self.placed[core].symbol
        outputs = {self.owner[(core, p.number)] for p in symbol.pins
                   if p.electrical_type == "output" and (core, p.number) in self.owner}
        inputs = {self.owner[(core, p.number)] for p in symbol.pins
                  if p.electrical_type == "input" and (core, p.number) in self.owner}
        feedback = [r for r in self.parts if r != core and len(self.placed[r].symbol.pins) == 2
                    and self.nets(r) & outputs and self.nets(r) & inputs]
        lanes = defaultdict(int)
        for r in self.parts:
            if r == core:
                continue
            nets = self.nets(r)
            if r in feedback:
                self.pair(r, min(nets & inputs), 100 + 26 * lanes["feedback"], 44, False)
                lanes["feedback"] += 1
            elif len(self.placed[r].symbol.pins) == 2 and nets & inputs and not nets & self.grounds:
                target = min((self.pin((core, p.number)) for p in symbol.pins
                              if self.owner.get((core, p.number)) in nets & inputs))
                first = min(nets - inputs) if nets - inputs else min(nets)
                self.pair(r, first, 50, target[1] + 18 * lanes["input"], False)
                if first not in self.grounds | self.supplies:
                    self.port_specs.append((first, (24, target[1] + 18 * lanes["input"])))
                lanes["input"] += 1
            elif len(self.placed[r].symbol.pins) == 2 and nets & self.grounds:
                self.pair(r, min(nets - self.grounds or nets), 146 + 24 * lanes["shunt"], 80, True)
                lanes["shunt"] += 1
            else:
                self.put(r, 50 + 30 * lanes["other"], 118)
                lanes["other"] += 1
        for i, r in enumerate(self.connectors):
            left = bool(self.nets(r) & inputs) and not bool(self.nets(r) & outputs)
            self.put(r, 20 if left else 194, 80 + i * 28, mirror="y" if left else None)
            if left:
                self.field_side[r] = -1
        if feedback:
            self.notes.append("feedback placed above amplifier from input/output pin connectivity")
        return True

    def place_connectors(self, x: float, y: float) -> None:
        for i, r in enumerate(self.connectors):
            self.put(r, x, y + 24 * i)

    def generated(self, name: str, net: str, at: Point, ref: str) -> PinKey:
        from ai_eda.compilers.schematic import _PlacedSymbol
        libref = self.library.resolve_symbol(LibraryRef(library="Connector" if name == "TestPoint" else "power", name=name))
        if not libref.verified:
            raise CompileError(f"textbook: requires installed power:{name}")
        symbol = self.library.load_symbol(libref)
        prov = Provenance(kind=ProvenanceKind.DERIVED, tool=self.compiler.id, tool_version=VERSION)
        c = Component(ref=ref, value=net if name != "PWR_FLAG" else name, symbol=libref, provenance=prov,
                      pins=[Pin(number=p.number, name=p.name, electrical_type=p.ir_type, provenance=prov) for p in symbol.pins])
        self.placed[ref] = _PlacedSymbol(c, symbol, "", mm(at[0]), mm(at[1]), Extent(0, 0, 0, 0))
        self.owner[ref, symbol.pins[0].number] = net
        return ref, symbol.pins[0].number

    def power(self) -> None:
        for i, (net, at) in enumerate(self.port_specs):
            self.generated("TestPoint", net, at, f"#PORT{i + 1:02}")
        pins = self.compiler._pin_map(self.placed)
        ground_keys = sorted((k for k in self.owner if self.owner[k] in self.grounds), key=lambda k: (natural_ref_key(k[0]), natural_ref_key(k[1])))
        for i, key in enumerate(ground_keys):
            p, pp = self.pin(key), pins[key]
            # Ground leaves in the library pin's outward direction, then down.
            end = (p[0] - round(pp.body_dir[0] * 6), p[1] - round(pp.body_dir[1] * 6))
            end = (end[0], max(end[1], p[1] + 8))
            ground = self.generated("GND", self.owner[key], end, f"#PWRG{i + 1:02}")
            self.ground_pairs.append((key, ground))
        for i, net in enumerate(sorted(self.supplies)):
            keys = [k for k in self.owner if self.owner[k] == net]
            if not keys:
                continue
            xs = [self.pin(k)[0] for k in keys]
            y = 22 - i * 10
            left, right = min(xs) - 8, max(xs) + 8
            self.line(net, (left, y), (right, y))
            self.generated("VCC", net, (left, y), f"#PWRS{i + 1:02}")
        # Only the new power-input glyphs need flags beyond the legacy policy.
        for i, net in enumerate(sorted(self.supplies | self.grounds)):
            keys = [k for k in self.owner if self.owner[k] == net]
            if not keys:
                continue
            types = {self.placed[k[0]].component.pin(k[1]).electrical_type.value for k in keys}
            if "power_out" not in types and types <= {"power_in", "input", "passive"}:
                key = next((k for k in keys if k[0].startswith("#PWRS")), next((k for k in keys if k[0].startswith("#PWRG")), keys[0]))
                p = self.pin(key)
                if net in self.supplies:
                    p = max((p for seg in self.segments[net] for p in seg), key=lambda p: p[0])
                at = (p[0] + 6, p[1])
                flag = self.generated("PWR_FLAG", net, at, f"#FLG{i + 1:02}")
                flag_ref = flag[0]
                others = [self.body_box(r) for r in self.placed if r != flag_ref]
                for offset in (6, -6, 12, -12, 18, -18):
                    self.put(flag_ref, p[0] + offset, p[1])
                    a = self.body_box(flag_ref)
                    if not any(self.overlap(a, b) for b in others):
                        break
                else:
                    raise CompileError(f"textbook: no clear power flag position on {net}")
                self.line(net, p, self.pin(flag))

    @staticmethod
    def overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
        return min(a[2], b[2]) - max(a[0], b[0]) > 1e-9 and min(a[3], b[3]) - max(a[1], b[1]) > 1e-9

    def body_box(self, ref: str) -> tuple[float, float, float, float]:
        ps = self.placed[ref]
        node = self.compiler._symbol_node(self.ir.project.id, sheet_uuid(self.ir.project.id), ps)
        out = render._Out()
        render._draw_symbol(out, [n for n in node if sexpr.head(n) != "property"], {ps.symbol.lib_id: ps.symbol.lib_symbols_entry()})
        return min(out.xs), min(out.ys), max(out.xs), max(out.ys)

    def nodes(self) -> tuple[list[Node], list[tuple[float, float, float, float]]]:
        """Upright fields beside measured symbol bounds, including pin text."""
        from ai_eda.compilers.schematic import _property
        nodes, boxes = [], []
        root = sheet_uuid(self.ir.project.id)
        for ref, ps in self.placed.items():
            node = self.compiler._symbol_node(self.ir.project.id, root, ps)
            body = [n for n in node if sexpr.head(n) != "property"]
            out = render._Out()
            render._draw_symbol(out, body, {ps.symbol.lib_id: ps.symbol.lib_symbols_entry()})
            box = (min(out.xs), min(out.ys), max(out.xs), max(out.ys))
            boxes.append(box)
            ps.extent = Extent(*box)
            for key in ("Reference", "Value"):
                node.remove(next(p for p in sexpr.find_all(node, "property") if p[1] == key))
                value = ref if key == "Reference" else ps.component.value
                if ref.startswith("#"):
                    field = _property(key, value, ps.x, snap(box[1] - 2.54), hide=key == "Reference" or not ref.startswith(("#PWRS", "#PORT")))
                else:
                    width = render._text_width(value, 1.27)
                    side = self.field_side.get(ref, 1)
                    x = box[2] + 2.54 + width / 2 if side == 1 else box[0] - 2.54 - width / 2
                    y = ps.y + (-1.27 if key == "Reference" else 1.27)
                    if len(ps.symbol.pins) == 2 and self.pin((ref, ps.symbol.pins[0].number))[1] == self.pin((ref, ps.symbol.pins[1].number))[1]:
                        x, y = ps.x, box[1] - (5.08 if key == "Reference" else 2.54)
                    field = _property(key, value, mm(round(x / GRID_MM)), mm(round(y / GRID_MM)), rot=ps.rotation % 180)
                node.append(field)
                field_out = render._Out()
                render._draw_fields(field_out, ["symbol", field], render._Instance(ps.x, ps.y, ps.rotation, ps.mirror))
                if field_out.xs:
                    field_box = (min(field_out.xs), min(field_out.ys), max(field_out.xs), max(field_out.ys))
                    boxes.append(field_box)
                    e = ps.extent
                    ps.extent = Extent(min(e.xmin, field_box[0]), min(e.ymin, field_box[1]), max(e.xmax, field_box[2]), max(e.ymax, field_box[3]))
            nodes.append(node)
        for a, b in combinations(boxes, 2):
            if self.overlap(a, b):
                raise CompileError(f"textbook: symbol/text bounds overlap: {a}, {b}")
        return nodes, boxes


def build(compiler: SchematicCompiler, ir: CircuitIR, library: KicadLibrary) -> tuple[Node, list[str]]:
    from ai_eda.compilers.schematic import GENERATOR, GENERATOR_VERSION, SCH_FORMAT_VERSION, _junction, _wire
    from ai_eda.compilers.schematic_textbook_route import route
    d = Drawing(compiler, ir, library)
    d.place()
    d.power()
    symbols, boxes = d.nodes()
    segments, junctions = route(d, boxes)
    pid = ir.project.id
    pins = compiler._pin_map(d.placed)
    stacked = compiler._stacks(d.placed, pins, d.owner)
    drawn = ir.model_copy(update={"nets": [n.model_copy(update={"pins": [PinRef(component_ref=k[0], pin_number=k[1]) for k in d.owner if d.owner[k] == n.name]}) for n in ir.nets]})
    no_connects, _, _ = compiler._connectivity(drawn, d.placed, pins, stacked)
    # The guard includes diagonal cross-coupling, unlike the legacy orthogonal guard.
    for net, segs in segments.items():
        for a, b in segs:
            for k, other_net in d.owner.items():
                if other_net != net and on(a, b, d.pin(k)):
                    raise CompileError(f"textbook wire {net} touches foreign pin {k}")
            for other_net, others in segments.items():
                if other_net != net and any(on(a, b, p) for seg in others for p in seg):
                    raise CompileError(f"textbook wire {net} touches a wire endpoint of {other_net}")
    wires = [_wire(tuple(map(mm, a)), tuple(map(mm, b)), net_item_uuid(pid, "textbook-wire", net, i)) for net, segs in sorted(segments.items()) for i, (a, b) in enumerate(segs)]
    dots = [_junction(tuple(map(mm, p)), net_item_uuid(pid, "textbook-junction", net, *p)) for net, pts in sorted(junctions.items()) for p in sorted(pts)]
    lib = {p.symbol.lib_id: p.symbol for p in d.placed.values()}
    return S("kicad_sch", S("version", SCH_FORMAT_VERSION), S("generator", Q(GENERATOR)), S("generator_version", Q(GENERATOR_VERSION)),
             S("uuid", Q(sheet_uuid(pid))), compiler._paper(d.placed),
             S("title_block", S("title", Q(ir.project.name)), S("comment", 1, Q(f"generated by {compiler.id} {VERSION} textbook from IR project {pid}"))),
             S("lib_symbols", *(lib[k].lib_symbols_entry() for k in sorted(lib))), *no_connects, *wires, *dots, *symbols,
             S("sheet_instances", S("path", Q("/"), S("page", Q("1")))), S("embedded_fonts", False)), ["layout: textbook 0.9; at most 20 parts and one IC; wired connections, no net labels", *d.notes]
