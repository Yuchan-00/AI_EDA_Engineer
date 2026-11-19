"""The textbook 1.0 stage driver: blocks, recipes, sheet variants, repair, routing, names, guards, the fallback ladder.

Stages: load and validate the model; draw each combination (two resistors that make one value) as one outline glyph
(:mod:`~ai_eda.compilers.textbook.compound`, put back as its two parts before the sheet is written and guarded again; a
layout failure with combinations draws the sheet again with every part on its own, noted, before the table rung);
partition it into blocks; draw each block with the first recipe of
:func:`~ai_eda.compilers.textbook.recipes.recipe_order` that claims it (a recipe that fails sends its block to the next,
with a note); arrange up to two sheet variants; per variant draw the rails, repair overlaps (at most
:data:`MAX_REPAIR_ROUNDS` rounds), route, name the nets, guard and score; keep the best. A hard finding left on the best
sheet redraws its blocks as block grid, once; one still left (or any :class:`LayoutFailure`) ends in the whole-sheet
table, the legacy table builder inside a 1.0 artifact, with a note. LayoutFailure never leaves :func:`build`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ai_eda.compilers.schematic_layout import Extent, layout_paper
from ai_eda.compilers.textbook import arrange as arrange_mod
from ai_eda.compilers.textbook import blocks as blocks_mod
from ai_eda.compilers.textbook import compound as compound_mod
from ai_eda.compilers.textbook import rails as rails_mod
from ai_eda.compilers.textbook import route as route_mod
from ai_eda.compilers.textbook.emit import all_segments, emit, guard
from ai_eda.compilers.textbook.glyphs import drawing_box, drawing_boxes, glyph_boxes, label_box, mm, overlaps, to_cells, union
from ai_eda.compilers.textbook.labels import name_nets
from ai_eda.compilers.textbook.model import BlockInfo, BoxMM, Drawing, DrawResult, Hit, Label, LayoutFailure, Score, Sheet, Wiring, load_model
from ai_eda.compilers.textbook.recipes import GRID, Ctx, Recipe, recipe_order
from ai_eda.ir import CircuitIR
from ai_eda.tools.kicad.library import KicadLibrary

VERSION = "1.0"
MAX_REPAIR_ROUNDS = 6
MAX_VARIANTS = 4  # per recipe draw
SHEET_VARIANTS = 2


@dataclass
class Evaluated:
    """One sheet variant routed, named, guarded and scored."""

    sheet: Sheet
    wiring: Wiring
    labels: list[Label]
    boxes: list[tuple[str, BoxMM]]
    hits: list[Hit]
    score: Score
    paper: tuple[str, float, float]
    notes: list[str]

    @property
    def hard(self) -> list[Hit]:
        return [h for h in self.hits if h.hard]


def build(compiler, ir: CircuitIR, library: KicadLibrary) -> DrawResult:
    """The textbook 1.0 drawing of ``ir``. IR and library faults raise CompileError; a layout reason never does."""
    original = load_model(compiler, ir, library)
    model, compounds = compound_mod.contract(original)
    ctx = Ctx(compiler, library, model, model.notes, original=original if compounds else None, compounds=compounds)
    fallbacks: list[str] = []
    try:
        return _textbook(ctx, fallbacks)
    except LayoutFailure as exc:
        if not compounds:
            return table_rung(ctx, fallbacks, exc.reason)
        reason = exc.reason
    # a combination that cannot be drawn as one element: every part on its own, before the whole-sheet table
    fallbacks = [f"combinations drawn as separate parts ({reason})"]
    model = load_model(compiler, ir, library)
    ctx = Ctx(compiler, library, model, model.notes)
    try:
        return _textbook(ctx, fallbacks)
    except LayoutFailure as exc:
        return table_rung(ctx, fallbacks, exc.reason)


def _textbook(ctx: Ctx, fallbacks: list[str]) -> DrawResult:
    model = ctx.model
    blocks = blocks_mod.partition(model)
    ctx.model = model = replace(model, blocks=blocks, rails=rails_mod.rail_order(model))
    order = recipe_order()
    drawn = {b.id: draw_block(ctx, b, order, fallbacks) for b in blocks}
    try:
        best: Evaluated | None = best_sheet(ctx, blocks, drawn)
        failure = None
    except LayoutFailure as exc:
        best, failure = None, exc
    if failure is not None or best.hard:
        if failure is not None:
            named = {failure.block} if failure.block else None
        else:
            named = _involved(best, blocks)
        guilty = [b.id for b in blocks if (named is None or b.id in named) and any(d.recipe != GRID.name for d in drawn[b.id])]
        if not guilty and failure is not None:
            raise failure
        if guilty:
            fallbacks.append(f"block grid for {', '.join(guilty)} ({failure.reason if failure is not None else best.hard[0]})")
            for block in blocks:
                if block.id in guilty:
                    drawn[block.id] = draw_block(ctx, block, (GRID,), fallbacks)
            best = best_sheet(ctx, blocks, drawn)
        if best.hard:
            raise LayoutFailure(f"{len(best.hard)} guard finding(s) left, first: {best.hard[0]}")
    ctx.notes.extend(best.notes)
    comment = f"generated by {ctx.compiler.id} {VERSION} textbook from IR project {model.project_id}"
    sheet, wiring, labels = best.sheet, best.wiring, best.labels
    out = ctx
    if ctx.compounds:  # each combination's outline drawn back as its two parts, checked again
        out = replace(ctx, model=replace(ctx.original, blocks=model.blocks, rails=model.rails))
        sheet, wiring, labels = compound_mod.expand(out.model, ctx.compounds, sheet, wiring, labels)
        hard = [h for h in guard(replace(out, notes=[]), sheet, wiring, labels, glyph_boxes(out.model, sheet)) if h.hard]
        if hard:
            raise LayoutFailure(f"combination drawn as two parts: {hard[0]}")
    node = emit(out, sheet, wiring, labels, best.paper, comment)
    recipes = ", ".join(f"{b.id}={'+'.join(d.recipe for d in drawn[b.id])}" for b in blocks)
    label_nets = {lab.net for lab in labels}
    head = (f"layout: textbook {VERSION}; blocks {recipes}; fallbacks: {'; '.join(fallbacks) or 'none'}; "
            f"{len(label_nets)} net(s) named by {len(labels)} global label(s) (DRC parity); paper {best.paper[0]}")
    spacing = [h for h in best.hits if not h.hard]
    tail = [f"spacing: {len(spacing)} finding(s), first: {spacing[0][len('spacing: '):]}"] if spacing else []
    return DrawResult(node, [head, *ctx.notes, *tail], tuple(fallbacks), best.score)


def draw_block(ctx: Ctx, block: BlockInfo, order: tuple[Recipe, ...], fallbacks: list[str]) -> list[Drawing]:
    """The drawings of one block: each recipe in ``order`` draws what it claims of the glyphs still free; a recipe that
    raises LayoutFailure leaves them to the next one, with a note."""
    free = set(block.members)
    out: list[Drawing] = []
    for recipe in order:
        while free:
            claimed = recipe.claim(ctx.model, block, frozenset(free))
            members = frozenset(claimed or ()) & free
            if not members:
                break
            try:
                variants = recipe.draw(ctx, block, members)[:MAX_VARIANTS]
                if not variants:
                    raise LayoutFailure(f"{recipe.name} returned no drawing", block.id)
            except LayoutFailure as exc:
                fallbacks.append(f"block {block.id}: {recipe.name} failed ({exc.reason}), next recipe")
                break
            ranked = [d if d.box is not None else replace(d, box=drawing_box(ctx.model, d)) for d in variants]
            out.append(min(ranked, key=lambda d: rank(ctx, d)) if len(ranked) > 1 else ranked[0])
            free -= members
        if not free:
            return out
    raise LayoutFailure(f"block {block.id}: {len(free)} glyph(s) no recipe drew", block.id)


def rank(ctx: Ctx, d: Drawing) -> tuple[int, int, int, int, int]:
    """Variant order inside a recipe: (internal faults, overlaps, sketch crossings, sketch length, box area)."""
    alone = Sheet((d,), ((0, 0),))
    boxes = drawing_boxes(ctx.model, d)
    wiring = Wiring(segments={n: list(s) for n, s in d.hints.items()})
    faults = sum(h.hard for h in guard(replace(ctx, notes=[]), alone, wiring, list(d.labels), boxes))
    x0, y0, x1, y1 = d.box or (0, 0, 0, 0)
    return faults, len(overlaps([b for _, b in boxes])), d.sketch[0], d.sketch[1], (x1 - x0) * (y1 - y0)


def best_sheet(ctx: Ctx, blocks: tuple[BlockInfo, ...], drawn: dict[str, list[Drawing]]) -> Evaluated:
    drawings = [d for b in blocks for d in drawn[b.id]]
    best: Evaluated | None = None
    for variant in range(SHEET_VARIANTS):
        sheet = arrange_mod.arrange(ctx, drawings, variant)
        if sheet is None:
            continue
        result = evaluate(ctx, sheet)
        if best is None or result.score < best.score:
            best = result
    if best is None:
        raise LayoutFailure("no sheet arrangement")
    return best


def evaluate(ctx: Ctx, sheet: Sheet) -> Evaluated:
    """Rails, repair, routing, names, guards and the score of one sheet variant (its notes kept apart until it wins)."""
    local = replace(ctx, notes=[])
    sheet = rails_mod.draw_rails(local, sheet)
    boxes = glyph_boxes(ctx.model, sheet)
    for _ in range(MAX_REPAIR_ROUNDS):
        pairs = overlaps([b for _, b in boxes])
        if not pairs:
            break
        repaired = arrange_mod.repair(local, sheet, [(boxes[i][0], boxes[j][0]) for i, j in pairs])
        if repaired is None:
            break
        sheet = repaired
        boxes = glyph_boxes(ctx.model, sheet)
    wiring = route_mod.route(local, sheet, boxes)
    labels = sheet.labels() + name_nets(local, sheet, wiring, boxes)
    hits = guard(local, sheet, wiring, labels, boxes)
    every = boxes + [("label", label_box(lab)) for lab in labels]
    segments = all_segments(wiring, labels)
    every += [("wire", (mm(min(a[0], b[0])), mm(min(a[1], b[1])), mm(max(a[0], b[0])), mm(max(a[1], b[1])))) for segs in segments.values() for a, b in segs]
    content = union([b for _, b in every])
    paper = layout_paper([Extent(*content)])
    return Evaluated(sheet, wiring, labels, boxes, hits, score(hits, wiring, labels, segments, content), paper, local.notes)


def score(hits: list[Hit], wiring: Wiring, labels: list[Label], segments: dict, content: BoxMM) -> Score:
    hard = sum(h.hard for h in hits)
    bends = 0
    for segs in segments.values():
        ends: dict = {}
        for a, b in segs:
            ends.setdefault(a, []).append((b[0] - a[0], b[1] - a[1]))
            ends.setdefault(b, []).append((a[0] - b[0], a[1] - b[1]))
        bends += sum(1 for dirs in ends.values() if len(dirs) == 2 and dirs[0][0] * dirs[1][1] != dirs[0][1] * dirs[1][0])
    wire = sum(abs(b[0] - a[0]) + abs(b[1] - a[1]) for segs in segments.values() for a, b in segs)
    x0, y0, x1, y1 = to_cells(content)
    return (hard, len(hits) - hard, wiring.crossings, len(labels) - len({lab.net for lab in labels}), bends, wire, (x1 - x0) * (y1 - y0))


def _involved(best: Evaluated, blocks: tuple[BlockInfo, ...]) -> set[str] | None:
    """The blocks the hard findings involve; None (every block) when a finding names nothing a drawing holds."""
    of = best.sheet.drawing_of()
    owners = [o for h in best.hard for o in h.owners]
    if any(not h.owners for h in best.hard) or any(o not in of for o in owners):
        return None
    return {best.sheet.drawings[of[o]].block for o in owners}


def table_rung(ctx: Ctx, fallbacks: list[str], reason: str) -> DrawResult:
    """The last rung: the legacy table builder in a 1.0 artifact (it needs no textbook geometry)."""
    compiler, ir = ctx.compiler, ctx.model.ir
    comment = f"generated by {compiler.id} {VERSION} textbook (table fallback) from IR project {ir.project.id}"
    node, notes = compiler._table(ir, ctx.library, comment)
    trail = [*fallbacks, "whole-sheet table"]
    head = f"layout: textbook {VERSION}; whole-sheet table fallback: {reason}; fallbacks: {'; '.join(trail)}"
    return DrawResult(node, [head, *notes], tuple(trail), None)
