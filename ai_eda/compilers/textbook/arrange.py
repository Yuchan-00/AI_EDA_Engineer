"""Arrangement: tiling the drawn blocks on one sheet.

:func:`arrange` builds sheet variant ``variant``. The spine sets the drawings left to right in block order (layer, then
declaration index), 8 cells apart, from the 25.4 mm origin. Vertically a drawing moves down only: so that its rail rows lie
on the rail lines the earlier drawings set, so that its first input port is level with the same net's port upstream, and
so that no rail line of another net runs over it; a drawing without rails (a connector block) moves down to the input it
feeds. A block whose only signal link is a host's bottom port is a satellite: it hangs below the host with its port under
the host's. When the spine is still wider than the usable width, pendant blocks (see :func:`pendants`) hang below the
block they feed too, widest first, so the supply lines stay one line across the top. A spine that is wider even then
wraps into bands 8 cells apart, cut at the block boundary that cuts the fewest signal nets (ties: the later boundary
that still fits); the rails are drawn again in each band, and a cut net is wired by the router or named by a label at
each end (label pairs that join two bands are noted). Each hung block gets a band of its own (its own rail lines), as
high as the drawings above its width allow. Variant 1 exists only when a block is a satellite by the bottom-port rule:
everything on the spine. The test points' block (:data:`~ai_eda.compilers.textbook.blocks.TEST_POINT_BLOCK`: pads named by
the labels or power symbols of their nets, no wire to any other block) is never on the spine: in every variant it hangs at the left, below
the drawings above it, in a band of its own.

The usable width is that of the smallest paper that holds the drawing: the A3 width (343.8 mm) first; when those bands
are taller than an A3 sheet, the drawing needs a larger paper anyway, so the bands are cut again at the A2 width (517.8
mm), then A1 - fewer, longer bands, the signal running on instead of back. A drawing that repeats one in an earlier band
(same recipe, same symbols: a second bank of voices) starts a band and stands in columns under it, like an array.

:func:`repair` widens the gap between two drawings whose boxes overlap by the overlap plus 2 cells and draws the rails
again."""

from __future__ import annotations

import math
from dataclasses import replace

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.compilers.textbook import blocks as blocks_mod
from ai_eda.compilers.textbook import rails as rails_mod
from ai_eda.compilers.textbook.glyphs import glyph_boxes
from ai_eda.compilers.textbook.model import Drawing, Model, Port, Role, Sheet, is_test_point

ORIGIN = 20  # cells: the content starts 25.4 mm from the sheet's left and top edges
BLOCK_GAP = 8  # cells (10.16 mm) between blocks
BAND_GAP = 8
USABLE = 270  # cells: the usable A3 width, 343.8 mm (420 - 25.4 - 50.8)
RAIL_HEAD = 8  # cells above a drawing's top rail row: the rail symbols and their names
RAIL_CLEAR = 3  # cells between a rail line and a drawing it passes over
ROUNDS = 4
A1_WIDTH_MM, A1_HEIGHT_MM = 841.0 - 50.8, 594.0 - 50.8  # the farthest content an A1 sheet holds (its right / bottom margin)
ORIGIN_MM, MARGIN_MM = 25.4, 50.8
WRAP_PAPERS = (("A3", 420.0, 297.0), ("A2", 594.0, 420.0), ("A1", 841.0, 594.0))  # the papers whose usable width cuts the bands
REPEAT_SLACK = 4  # cells two repeated drawings' widths may differ by (their field texts) and still be column-aligned


def _note(notes: list[str], text: str) -> None:
    """A note of this sheet variant, once; it travels with the sheet (``Sheet.notes``) and reaches the build only when
    the variant is kept (both variants are arranged before one is)."""
    if text not in notes:
        notes.append(text)


def _box(d: Drawing) -> tuple[int, int, int, int]:
    return d.box or (0, 0, 0, 0)


def _top(d: Drawing) -> int:
    return min([_box(d)[1]] + [s.y - RAIL_HEAD for s in d.rails])


def cut(model: Model, drawings: list[Drawing], units: list[int], width: int = USABLE, lead: dict[int, int] | None = None) -> list[list[int]]:
    """The units (drawing indices, spine order) in bands no wider than ``width`` cells where possible: the fewest bands, then
    the fewest repeats (``lead``: unit -> the cells it moves right to stand under the drawing it repeats) not at the start
    of a band, then the fewest signal nets cut over all cuts, ties to the later boundaries (chosen over the whole spine,
    not band by band, so a cut that is free early never leaves a narrow band behind it). A band that starts with a repeat
    is wider by its lead. A block wider than ``width`` gets a band alone."""
    lead = lead or {}

    def span(us: list[int]) -> int:
        return sum(_box(drawings[i])[2] - _box(drawings[i])[0] for i in us) + BLOCK_GAP * (len(us) - 1) + lead.get(us[0], 0)

    def nets(us: list[int]) -> set[str]:
        return {n for i in us for k in drawings[i].place if not is_test_point(model.glyphs[k]) for n in model.glyphs[k].net.values()
                if model.nets[n].role is Role.SIGNAL}

    rest = list(units)
    if not rest or span(rest) <= width and not any(u in lead for u in rest[1:]):
        return [rest] if rest else []
    ends = [j for j in range(1, len(rest)) if drawings[rest[j]].block != drawings[rest[j - 1]].block] or list(range(1, len(rest)))
    stops = [0, *ends, len(rest)]
    cost = {j: len(nets(rest[:j]) & nets(rest[j:])) for j in ends}
    best: dict[int, tuple] = {0: (0, 0, 0, ())}
    back: dict[int, int] = {}
    for b, i in enumerate(stops[1:], 1):
        for j in stops[:b]:
            if j not in best:
                continue
            single = len({drawings[u].block for u in rest[j:i]}) == 1
            if span(rest[j:i]) > width and not single:
                continue
            bands, inner, cuts, where = best[j]
            key = (bands + 1, inner + sum(u in lead for u in rest[j + 1:i]), cuts + cost.get(i, 0), (*where, -i))
            if i not in best or key < best[i]:
                best[i], back[i] = key, j
    if len(rest) not in best:
        return [rest]
    out: list[list[int]] = []
    i = len(rest)
    while i:
        j = back[i]
        out.append(rest[j:i])
        i = j
    return out[::-1]


def _ports(model: Model, drawings: list[Drawing]) -> dict[str, tuple[Port, ...]]:
    out: dict[str, tuple[Port, ...]] = {}
    for b in model.blocks:
        ds = [d for d in drawings if d.block == b.id]
        out[b.id] = tuple(p for d in ds for p in (d.ports or blocks_mod.ports(model, b, d)))
    return out


def _width(drawings: list[Drawing], units: list[int]) -> int:
    return sum(_box(drawings[i])[2] - _box(drawings[i])[0] for i in units) + BLOCK_GAP * max(len(units) - 1, 0)


def pendants(model: Model, drawings: list[Drawing], spine: list[int], usable: int = USABLE) -> dict[str, tuple[str, Port]]:
    """Blocks that leave the spine when it is wider than the usable width: a block (not only connectors) whose only signal
    link is one of its outputs that one other block takes as an input hangs below that block, widest first, until the
    spine fits."""
    ports = _ports(model, drawings)
    signal = {b.id: {n for k in b.members if not is_test_point(model.glyphs[k]) for n in model.glyphs[k].net.values()
                     if model.nets[n].role is Role.SIGNAL} for b in model.blocks}
    blocks = {b.id: b for b in model.blocks}
    out: dict[str, tuple[str, Port]] = {}
    width = {b: _width(drawings, [i for i in spine if drawings[i].block == b]) for b in blocks}
    units = list(spine)
    for b in sorted(blocks.values(), key=lambda b: (-width[b.id], b.id)):
        if _width(drawings, units) <= usable:
            break
        if blocks_mod.connectors_only(model, b) or not any(drawings[i].block == b.id for i in units):
            continue
        links = sorted(n for n in signal[b.id] if any(n in signal[o] for o in signal if o != b.id))
        if len(links) != 1 or links[0] not in b.outputs:
            continue
        hosts = [o for o in blocks.values() if o.id != b.id and links[0] in o.inputs and o.id not in out and not blocks_mod.connectors_only(model, o)]
        port = next((q for q in ports.get(hosts[0].id, ()) if q.net == links[0]), None) if len(hosts) == 1 else None
        if port is None or hosts[0].id in out:
            continue
        out[b.id] = (hosts[0].id, port)
        units = [i for i in units if drawings[i].block != b.id]
    return out if _width(drawings, units) < _width(drawings, spine) else {}


def arrange(ctx, drawings: list[Drawing], variant: int) -> Sheet | None:
    """Sheet variant ``variant``, its bands cut at the usable width of the smallest paper that holds it: A3 first; when the
    A3 bands are taller than an A3 sheet the drawing needs the larger paper anyway, so the bands are cut again at its
    width (fewer, longer bands, the signal running on instead of back), and so on up to A1."""
    model: Model = ctx.model
    if variant > 1:
        return None
    sheet = None
    for name, width, height in WRAP_PAPERS:
        usable = int((width - ORIGIN_MM - MARGIN_MM) / 1.27)
        sheet = _arrange(model, drawings, variant, usable, width - ORIGIN_MM - MARGIN_MM)
        if sheet is None:
            return None
        right, bottom = _extent(sheet)
        if right * 1.27 + MARGIN_MM <= width and bottom * 1.27 + MARGIN_MM <= height:
            break
    return sheet


def _extent(sheet: Sheet) -> tuple[int, int]:
    boxes = [(_box(d)[2] + dx, _box(d)[3] + dy) for d, (dx, dy) in zip(sheet.drawings, sheet.offset)]
    return max((b[0] for b in boxes), default=0), max((b[1] for b in boxes), default=0)


def _arrange(model: Model, drawings: list[Drawing], variant: int, usable: int, usable_mm: float) -> Sheet | None:
    order = {b.id: i for i, b in enumerate(model.blocks)}
    every = sorted(range(len(drawings)), key=lambda i: (order.get(drawings[i].block, len(order)), i))
    pads = {d.block for d in drawings if d.block == blocks_mod.TEST_POINT_BLOCK}  # hung below the content (module docstring)
    sats = blocks_mod.satellites(model, model.blocks, _ports(model, drawings)) if model.blocks else {}
    sats = {k: v for k, v in sats.items() if k not in pads}
    rest = [i for i in every if drawings[i].block not in sats and drawings[i].block not in pads]
    extra = pendants(model, drawings, rest, usable) if _width(drawings, rest) > usable else {}
    if variant == 1 and not sats:
        return None
    sats = {} if variant == 1 else {**sats, **extra}  # variant 1: everything on the spine
    spine = [i for i in every if drawings[i].block not in sats and drawings[i].block not in pads]
    notes: list[str] = []
    bands = cut(model, drawings, spine, usable)
    offsets = _place_bands(drawings, bands)
    lead = _leads(model, drawings, bands, offsets)
    if lead:  # a repeat to stand under the drawing it repeats: cut again, so it starts a band with room to move right
        bands = cut(model, drawings, spine, usable, lead)
        offsets = _place_bands(drawings, bands)
    if len(bands) > 1:
        _note(notes, f"bands: the spine is wider than {usable} cells ({usable_mm:.1f} mm); wrapped into {len(bands)} bands")
    _align_repeats(model, drawings, bands, offsets, notes, usable)
    paired: dict[str, set[int]] = {}
    for b, band in enumerate(bands):
        for i in band:
            for lab in drawings[i].labels:
                if lab.kind == "pair":
                    paired.setdefault(lab.net, set()).add(b)
    across = sorted((n for n, bs in paired.items() if len(bs) > 1), key=natural_ref_key)
    if across:
        _note(notes, f"label pairs: {', '.join(across)} join drawings in different bands; each is named at both ends, not wired across the bands")
    bands_out = [list(b) for b in bands]
    for i in every:
        d = drawings[i]
        if d.block not in sats and d.block not in pads:
            continue
        dx = ORIGIN - _box(d)[0]
        hosts, own = [], None
        if d.block in pads:
            _note(notes, f"block {d.block}: hung at the left below the content, each pad on a label or power symbol of its net")
        else:
            host, port = sats[d.block]
            hosts = [j for j in spine if drawings[j].block == host and any(p.net == port.net for p in drawings[j].ports)]
            own = next((p for p in d.ports if p.net == port.net), None)
        if hosts and own is not None:
            j = hosts[0]
            hx = next(p for p in drawings[j].ports if p.net == port.net).at[0] + offsets[j][0]
            dx = max(hx - own.at[0] - (2 if port.side == "L" else 0), ORIGIN - _box(d)[0])
            _note(notes, f"block {d.block}: hung below {host}, whose {port.net} it feeds")
        x0, x1 = _box(d)[0] + dx, _box(d)[2] + dx
        under = [_box(drawings[j])[3] + offsets[j][1] for j in offsets if _box(drawings[j])[0] + offsets[j][0] < x1 + 2
                 and x0 < _box(drawings[j])[2] + offsets[j][0] + 2]
        _place_band(drawings, [i], max(under, default=ORIGIN) + BAND_GAP, offsets)
        offsets[i] = (dx, offsets[i][1])
        bands_out.append([i])
    right = max((_box(drawings[i])[2] + offsets[i][0] for i in offsets), default=0)
    bottom = max((_box(drawings[i])[3] + offsets[i][1] for i in offsets), default=0)
    if right * 1.27 > A1_WIDTH_MM or bottom * 1.27 > A1_HEIGHT_MM:
        _note(notes, "paper: the drawing is larger than A1; it stays one sheet (splitting the design into sheets would help)")
    return Sheet(tuple(drawings), tuple(offsets[i] for i in range(len(drawings))), bands=tuple(tuple(b) for b in bands_out), notes=tuple(notes))


def _place_bands(drawings: list[Drawing], bands: list[list[int]]) -> dict[int, tuple[int, int]]:
    offsets: dict[int, tuple[int, int]] = {}
    top = ORIGIN
    for band in bands:
        _place_band(drawings, band, top, offsets)
        top = max(_box(drawings[i])[3] + offsets[i][1] for i in band) + BAND_GAP + RAIL_HEAD
    return offsets


def _repeats(model: Model, drawings: list[Drawing], order: list[int]) -> dict[int, int]:
    """Drawing -> the earlier drawing it repeats (same recipe, same symbols, widths within :data:`REPEAT_SLACK` cells)."""
    first: dict[tuple, int] = {}
    out: dict[int, int] = {}
    for i in order:
        if len(drawings[i].place) < 2:
            continue
        sig = _signature(model, drawings[i])
        j = first.setdefault(sig, i)
        if j != i and abs((_box(drawings[i])[2] - _box(drawings[i])[0]) - (_box(drawings[j])[2] - _box(drawings[j])[0])) <= REPEAT_SLACK:
            out[i] = j
    return out


def _leads(model: Model, drawings: list[Drawing], bands: list[list[int]], offsets: dict[int, tuple[int, int]]) -> dict[int, int]:
    """Repeat -> how far right of a band's start it must stand to be under the drawing it repeats (placed as ``offsets``)."""
    order = [i for b in bands for i in b]
    out = {}
    for i, j in _repeats(model, drawings, order).items():
        if not any(i in b and j in b for b in bands):
            out[i] = max(0, _box(drawings[j])[0] + offsets[j][0] - ORIGIN)
    return out


def _signature(model: Model, d: Drawing) -> tuple:
    """What makes two drawings repeats of one another: the recipe and the library symbols of their glyphs (unit by unit). A
    combination's outline counts as one of its parts: a value made of two resistors stands where one resistor would."""
    return d.recipe, tuple(sorted((model.outlines.get(k, model.glyphs[k].symbol.lib_id), model.glyphs[k].unit) for k in d.place))


def _align_repeats(model: Model, drawings: list[Drawing], bands: list[list[int]], offsets: dict[int, tuple[int, int]],
                   notes: list[str], usable: int = USABLE) -> None:
    """Repeated drawings in different bands stand in columns, like an array: a drawing that repeats one in an earlier band
    moves right so its left edge is under that one's, with the drawings after it in its band; never left (nothing in its
    band comes closer), and not when its band would then be wider than ``usable``."""
    order = [i for b in bands for i in b]
    for i, j in _repeats(model, drawings, order).items():
        band = next(b for b in bands if i in b)
        if j in band:
            continue
        delta = (_box(drawings[j])[0] + offsets[j][0]) - (_box(drawings[i])[0] + offsets[i][0])
        right = max(_box(drawings[k])[2] + offsets[k][0] for k in band) + max(delta, 0)
        if delta <= 0 or right - ORIGIN > usable:
            continue
        for k in band[band.index(i):]:
            offsets[k] = (offsets[k][0] + delta, offsets[k][1])
        _note(notes, f"block {drawings[i].block}: column-aligned under block {drawings[j].block}, which it repeats")


def _place_band(drawings: list[Drawing], band: list[int], top: int, offsets: dict[int, tuple[int, int]]) -> None:
    """Left to right from the origin, each drawing moved down only: onto the rail lines, level with its upstream port,
    clear of rail lines that pass over it."""
    x = ORIGIN
    for i in band:
        d = drawings[i]
        offsets[i] = (x - _box(d)[0], top - _top(d))
        x += _box(d)[2] - _box(d)[0] + BLOCK_GAP
    for _ in range(ROUNDS):
        rail_y: dict[str, int] = {}
        out_y: dict[str, int] = {}
        moved = False
        for i in band:
            d = drawings[i]
            dx, dy = offsets[i]
            need = top - _top(d)
            for s in d.rails:
                if s.net in rail_y:
                    need = max(need, rail_y[s.net] - s.y)
            main = next((p for p in d.ports if p.role == "in" and p.net in out_y), None)
            if main is not None:
                need = max(need, out_y[main.net] - main.at[1])
            if need > dy:
                offsets[i] = (dx, need)
                moved = True
            dy = offsets[i][1]
            for s in d.rails:
                rail_y.setdefault(s.net, s.y + dy)
            for p in d.ports:
                if p.role in ("out", "slot"):
                    out_y.setdefault(p.net, p.at[1] + dy)
        # a rail line runs from its first to its last attachment in the band: a drawing under it that does not attach moves down
        extent: dict[str, tuple[int, int]] = {}
        for i in band:
            dx = offsets[i][0]
            for s in drawings[i].rails:
                lo, hi = extent.get(s.net, (s.x0 + dx, s.x1 + dx))
                extent[s.net] = (min(lo, s.x0 + dx), max(hi, s.x1 + dx + rails_mod.GLYPH_STUB))
        for i in band:
            d = drawings[i]
            dx, dy = offsets[i]
            x0, y0, x1, _ = _box(d)
            own = {s.net for s in d.rails}
            for net, (lo, hi) in sorted(extent.items()):
                y = rail_y.get(net)
                if net in own or y is None or hi < x0 + dx or lo > x1 + dx:
                    continue
                if y > y0 + dy - RAIL_CLEAR and y < _box(d)[3] + dy + RAIL_CLEAR:
                    offsets[i] = (dx, y + RAIL_CLEAR - y0)
                    dy = offsets[i][1]
                    moved = True
        # an upstream drawing without rails moves down to the input port it feeds
        for i in reversed(band):
            d = drawings[i]
            if d.rails:
                continue
            dx, dy = offsets[i]
            for p in d.ports:
                if p.role not in ("out", "slot"):
                    continue
                match = next(((j, q) for j in band[band.index(i) + 1:] for q in drawings[j].ports if q.net == p.net and q.role == "in"), None)
                if match is None:
                    continue
                j, q = match
                want = q.at[1] + offsets[j][1] - p.at[1]
                if want > dy:
                    offsets[i] = (dx, want)
                    moved = True
                break
        if not moved:
            break


def repair(ctx, sheet: Sheet, overlaps: list[tuple[str, str]]) -> Sheet | None:
    """Two drawings whose boxes overlap: the gap before the right one (and everything after it in its band) widens by the
    overlap plus 2 cells; the rails are drawn again. None when no overlap is between two drawings."""
    model: Model = ctx.model
    of = sheet.drawing_of()
    boxes: dict[str, list] = {}
    for tag, box in glyph_boxes(model, sheet):
        boxes.setdefault(tag, []).append(box)
    shift: dict[int, int] = {}
    for a, b in overlaps:
        ia, ib = of.get(_owner(a)), of.get(_owner(b))
        if ia is None or ib is None or ia == ib:
            continue
        right = max((ia, ib), key=lambda i: (_box(sheet.drawings[i])[0] + sheet.offset[i][0], i))
        width = max((min(p[2], q[2]) - max(p[0], q[0]) for p in boxes.get(a, ()) for q in boxes.get(b, ())), default=0.0)
        need = math.ceil(width / 1.27) + 2
        shift[right] = max(shift.get(right, 0), need)
    if not shift:
        return None
    bands = sheet.bands or (tuple(range(len(sheet.drawings))),)
    offsets = list(sheet.offset)
    for right, need in sorted(shift.items()):
        band = next((b for b in bands if right in b), ())
        start = _box(sheet.drawings[right])[0] + sheet.offset[right][0]
        for i in band:
            if _box(sheet.drawings[i])[0] + sheet.offset[i][0] >= start:
                offsets[i] = (offsets[i][0] + need, offsets[i][1])
    ctx.notes.append(f"repair: widened {len(shift)} gap(s) between overlapping blocks")
    return rails_mod.draw_rails(ctx, replace(sheet, offset=tuple(offsets), glyphs=(), links=(), rails=()))


def _owner(tag: str) -> str:
    return tag.split(":", 1)[1] if not tag.startswith("label:") else tag
