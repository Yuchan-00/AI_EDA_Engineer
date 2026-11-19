"""Recipes: how one block (or part of one) is drawn, and the block grid every block can fall back to.

A recipe claims a set of glyphs of a block (``claim`` returns them, or None)
and draws them (``draw`` returns up to four block-relative variants, or
raises :class:`~ai_eda.compilers.textbook.model.LayoutFailure`, and the next
recipe is tried with a note). :data:`RECIPE_ORDER` is the order they are
tried in; the block grid comes last and claims whatever is left.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

from ai_eda.compilers.schematic_layout import LAYOUT_GAP_MM, Extent, layout_columns, layout_positions, natural_ref_key
from ai_eda.compilers.textbook.glyphs import body_box, cell, field_anchors, label_box, measure, mm, pin_cell, pin_side, to_cells, union
from ai_eda.compilers.textbook.labels import stub_label
from ai_eda.compilers.textbook.model import BlockInfo, BoxMM, Drawing, Generated, Label, Model, Orient, Place, Pt, Role, Seg, Side
from ai_eda.tools.kicad.library import KicadLibrary

if TYPE_CHECKING:
    from ai_eda.compilers.schematic import SchematicCompiler

#: a net with more pins than this is named by labels at its pins in the block grid, not left to the router
GRID_WIRED_PINS = 4
#: field sides the block grid tries, in order, when a side has no connected pin
FIELD_SIDES: tuple[Side, ...] = ("R", "T", "B", "L")


@dataclass
class Ctx:
    """What a recipe may use: the compiler (node builders, checks), the library, the model and the one notes list."""

    compiler: SchematicCompiler
    library: KicadLibrary
    model: Model
    notes: list[str]
    budget: int = 60_000  # router expansions per search
    original: Model | None = None  # the model before combinations were drawn as one outline each (None: none were)
    compounds: dict = field(default_factory=dict)  # outline key -> compound.Compound


class Recipe(Protocol):
    name: str

    def claim(self, model: Model, block: BlockInfo, free: frozenset[str]) -> frozenset[str] | None: ...

    def draw(self, ctx: Ctx, block: BlockInfo, members: frozenset[str]) -> list[Drawing]: ...


@dataclass(frozen=True)
class _Cell:
    key: str
    place: Place  # at the cell origin (0, 0)
    stubs: dict[str, list[Seg]]
    labels: list[Label]
    anchors: tuple[Pt, Pt]
    extent: BoxMM


class GridRecipe:
    """The block grid: the block's glyphs unturned in a table whose column widths and row heights are measured (body, pin
    stubs, the label at every stub end, the fields). Every connected pin gets a 2-cell stub; a net that leaves the block
    or has more than :data:`GRID_WIRED_PINS` pins gets a global label at each of its pins here, smaller inner nets are left
    to the router (and to name labels where it draws nothing)."""

    name = "grid"

    def claim(self, model: Model, block: BlockInfo, free: frozenset[str]) -> frozenset[str] | None:
        return frozenset(free) or None

    def draw(self, ctx: Ctx, block: BlockInfo, members: frozenset[str]) -> list[Drawing]:
        model = ctx.model
        keys = sorted(members, key=natural_ref_key)
        cells = {key: self._cell(model, key) for key in keys}
        extents = {key: Extent(*cells[key].extent) for key in keys}
        positions = layout_positions(extents, origin=(0.0, 0.0), columns=self._columns(len(keys)), gap=LAYOUT_GAP_MM, min_pitch=0.0, order=keys)
        inside = set(members)
        place: dict[str, Place] = {}
        hints: dict[str, list[Seg]] = {}
        labels: list[Label] = []
        fields: dict[str, tuple[Pt, Pt]] = {}
        boxes: list[BoxMM] = []
        for key in keys:
            c = cells[key]
            dx, dy = cell(positions[key][0]), cell(positions[key][1])
            place[key] = c.place.moved(dx, dy)
            for net, segs in c.stubs.items():
                hints.setdefault(net, []).extend(((a[0] + dx, a[1] + dy), (b[0] + dx, b[1] + dy)) for a, b in segs)
            labels += [lab.moved(dx, dy) for lab in c.labels if _labelled(model, lab.net, inside)]
            (rx, ry), (vx, vy) = c.anchors
            fields[key] = ((rx + dx, ry + dy), (vx + dx, vy + dy))
            x0, y0, x1, y1 = c.extent
            boxes.append((x0 + mm(dx), y0 + mm(dy), x1 + mm(dx), y1 + mm(dy)))
        return [Drawing(block=block.id, recipe=self.name, place=place, hints={n: tuple(s) for n, s in sorted(hints.items())},
                        labels=tuple(labels), fields=fields, box=to_cells(union(boxes)))]

    @staticmethod
    def _columns(count: int) -> int:
        return layout_columns(count)

    @staticmethod
    def _cell(model: Model, key: str) -> _Cell:
        g = model.glyphs[key]
        o = Orient()
        stubs: dict[str, list[Seg]] = {}
        labels: list[Label] = []
        used: dict[str, int] = {s: 0 for s in FIELD_SIDES}
        for number in sorted(g.net, key=natural_ref_key):
            if (g.ref, number) in model.stacked:
                continue  # drawn at its representative's point
            side = pin_side(g, number, o)
            lab = stub_label(g.net[number], pin_cell(g, number, (0, 0), o), side)
            stubs.setdefault(lab.net, []).append(lab.stub)  # type: ignore[arg-type]
            labels.append(lab)
            used[side] += 1
        body = body_box(g, Place(0, 0, o))
        around = union([body, *(label_box(lab) for lab in labels)])
        side = next((s for s in FIELD_SIDES if not used[s]), min(FIELD_SIDES, key=lambda s: used[s]))
        place = Place(0, 0, o, side)
        m = measure(g, place, around=around)
        return _Cell(key, place, stubs, labels, field_anchors(m.field_nodes), union([around, *m.fields]))


def _labelled(model: Model, net: str, inside: set[str]) -> bool:
    """A net the block grid names at its pins: it leaves the block, or has more than GRID_WIRED_PINS drawn pins."""
    info = model.nets[net]
    drawn = [(k, n) for k, n in info.pins if (model.glyphs[k].ref, n) not in model.stacked]
    return any(k not in inside for k, _ in info.pins) or len(drawn) > GRID_WIRED_PINS


GRID = GridRecipe()


class TestPointRecipe(GridRecipe):
    """The test points' block (:data:`~ai_eda.compilers.textbook.blocks.TEST_POINT_BLOCK`): each pad drawn at its net by name,
    as the rest of the sheet names that net - a pad on a ground (negative) net with the net's power symbol hanging below
    its stub, a pad on a supply turned pin-up under the supply's power symbol, a pad on a signal net with a 2-cell stub and a
    global label of the net (as the block grid draws a pin whose net leaves the block) - with no wire to route, in one row
    (a strip under the content, as low as a row of pads can be). Claims a block made only of test points, nothing else."""

    name = "test_points"

    @staticmethod
    def _columns(count: int) -> int:
        return count

    def claim(self, model: Model, block: BlockInfo, free: frozenset[str]) -> frozenset[str] | None:
        from ai_eda.compilers.textbook.blocks import is_test_point

        return frozenset(free) if free and all(is_test_point(model, k) for k in free) else None

    def draw(self, ctx: Ctx, block: BlockInfo, members: frozenset[str]) -> list[Drawing]:
        model = ctx.model
        keys = sorted(members, key=natural_ref_key)
        cells: dict[str, _Cell] = {}
        symbols: dict[str, Generated] = {}
        for key in keys:
            cells[key], gen = self._pad_cell(ctx, key)
            if gen is not None:
                symbols[key] = gen
        extents = {key: Extent(*cells[key].extent) for key in keys}
        positions = layout_positions(extents, origin=(0.0, 0.0), columns=len(keys), gap=LAYOUT_GAP_MM, min_pitch=0.0, order=keys)
        place: dict[str, Place] = {}
        hints: dict[str, list[Seg]] = {}
        labels: list[Label] = []
        glyphs: list[Generated] = []
        fields: dict[str, tuple[Pt, Pt]] = {}
        boxes: list[BoxMM] = []
        for key in keys:
            c = cells[key]
            dx, dy = cell(positions[key][0]), cell(positions[key][1])
            place[key] = c.place.moved(dx, dy)
            for net, segs in c.stubs.items():
                hints.setdefault(net, []).extend(((a[0] + dx, a[1] + dy), (b[0] + dx, b[1] + dy)) for a, b in segs)
            labels += [lab.moved(dx, dy) for lab in c.labels]
            if key in symbols:
                glyphs.append(replace(symbols[key].moved(dx, dy), ref=f"#PWR{len(glyphs) + 1:02d}"))
            (rx, ry), (vx, vy) = c.anchors
            fields[key] = ((rx + dx, ry + dy), (vx + dx, vy + dy))
            x0, y0, x1, y1 = c.extent
            boxes.append((x0 + mm(dx), y0 + mm(dy), x1 + mm(dx), y1 + mm(dy)))
        return [Drawing(block=block.id, recipe=self.name, place=place, hints={n: tuple(s) for n, s in sorted(hints.items())},
                        glyphs=tuple(glyphs), labels=tuple(labels), fields=fields, box=to_cells(union(boxes)))]

    @classmethod
    def _pad_cell(cls, ctx: Ctx, key: str) -> tuple[_Cell, Generated | None]:
        """One pad's cell: a signal net's as the block grid draws it; a rail net's with its power symbol on the stub."""
        from ai_eda.compilers.textbook import rails as rails_mod
        from ai_eda.compilers.textbook.glyphs import generated_boxes

        model = ctx.model
        g = model.glyphs[key]
        number = min(g.net, key=natural_ref_key)  # a test point has one pin
        net = g.net[number]
        role = model.nets[net].role
        if role is Role.SIGNAL:
            return cls._cell(model, key), None
        o = Orient(180) if role is Role.SUPPLY else Orient()  # a supply's symbol stands above its pin
        kind = "GND" if role is Role.GROUND else "NEG" if role is Role.NEGATIVE else "VCC"
        gen, stubs = rails_mod.local_glyph(ctx.library, kind, net, pin_cell(g, number, (0, 0), o), pin_side(g, number, o), "#PWR01")
        body, symbol_fields = generated_boxes(gen)
        around = union([body_box(g, Place(0, 0, o)), body, *symbol_fields])
        place = Place(0, 0, o, "R")
        m = measure(g, place, around=around)
        return _Cell(key, place, {net: list(stubs)}, [], field_anchors(m.field_nodes), union([around, *m.fields])), gen


TEST_POINTS = TestPointRecipe()


def recipe_order() -> tuple[Recipe, ...]:
    """The recipes in the order a block tries them: the test points' block, motifs (hub, op-amp stage, cross-coupled pair,
    array, supply chain), then the general columns recipe, then the block grid."""
    from ai_eda.compilers.textbook import columns, hub, motifs

    return (TEST_POINTS, hub.HUB, motifs.OPAMP_STAGE, motifs.CROSS_COUPLED, motifs.ARRAY, motifs.SUPPLY_CHAIN, columns.COLUMNS, GRID)


def __getattr__(name: str):  # RECIPE_ORDER is resolved on first use: the recipe modules import this one
    if name == "RECIPE_ORDER":
        return recipe_order()
    raise AttributeError(name)
