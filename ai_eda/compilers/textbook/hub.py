"""The IC hub: a large IC (20 pins or more) at rotation 0 with its headers planned around it.

:func:`header_plans` is the header fan-out plan of design section 7: a header is a connector whose signal pins mostly
reach the IC's signal pins (same net); its side is the IC side holding most of the pins it reaches, never its block I/O;
on that side its pins are matched to the IC's pin rows by the longest increasing subsequence (LIS) of the IC rows taken
in header pin order, trying the header as drawn and flipped (a tie keeps it as drawn, an equal subsequence the first
one found); the LIS pins are wired by a monotone channel (no crossing by construction), the rest are named by label
pairs. The vertical offset that lines the header up is the median of (IC row - header row) over the LIS pins.

``HUB.claim`` takes nothing yet: placing a header beside the IC is a sheet arrangement across blocks (the headers are
blocks of their own in the corpus), so until the arrangement takes the plan the block grid draws the IC and the notes
say so.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.compilers.textbook.model import BlockInfo, Drawing, Glyph, Kind, LayoutFailure, Model, Role, Side

HUB_PINS = 20


@dataclass(frozen=True)
class HeaderPlan:
    """One header's fan-out: the IC side it faces, whether it is flipped, the (header pin, IC pin) pairs wired by the
    channel (LIS), the pins left to label pairs, and the vertical offset (cells) that lines it up."""

    header: str
    side: Side
    flipped: bool
    wired: tuple[tuple[str, str], ...]
    labelled: tuple[tuple[str, str], ...]
    offset: int

    @property
    def reached(self) -> int:
        return len(self.wired) + len(self.labelled)


def hubs(model: Model) -> list[Glyph]:
    return sorted((g for g in model.glyphs.values() if g.kind is Kind.IC and len(g.pins) >= HUB_PINS), key=lambda g: natural_ref_key(g.key))


def lis(seq: list[int]) -> list[int]:
    """Indices of a longest strictly increasing subsequence of ``seq`` (the first such in patience order)."""
    tails: list[int] = []
    tail_at: list[int] = []
    back = [-1] * len(seq)
    for i, v in enumerate(seq):
        k = bisect.bisect_left(tails, v)
        if k == len(tails):
            tails.append(v)
            tail_at.append(i)
        else:
            tails[k] = v
            tail_at[k] = i
        back[i] = tail_at[k - 1] if k else -1
    out = []
    i = tail_at[-1] if tail_at else -1
    while i >= 0:
        out.append(i)
        i = back[i]
    return out[::-1]


def _ic_rows(model: Model, ic: Glyph) -> dict[str, tuple[Side, int, str]]:
    """Signal net -> (IC side, row along that side, IC pin) of its first IC pin."""
    out: dict[str, tuple[Side, int, str]] = {}
    for p in ic.pins:
        if p.number in ic.net and model.nets[ic.net[p.number]].role is Role.SIGNAL:
            out.setdefault(ic.net[p.number], (p.side, p.dy if p.side in ("L", "R") else p.dx, p.number))
    return out


def header_plans(model: Model, ic: Glyph) -> list[HeaderPlan]:
    """The fan-out plan of every header of ``ic`` (see the module text), headers in natural order."""
    rows = _ic_rows(model, ic)
    plans = []
    for g in sorted((g for g in model.glyphs.values() if g.kind is Kind.CONN), key=lambda g: natural_ref_key(g.key)):
        sig = [p for p in g.pins if p.number in g.net and model.nets[g.net[p.number]].role is Role.SIGNAL]
        reach = [p for p in sig if g.net[p.number] in rows]
        if not sig or 2 * len(reach) <= len(sig):
            continue
        count: dict[str, int] = {}
        for p in reach:
            s = rows[g.net[p.number]][0]
            count[s] = count.get(s, 0) + 1
        side = max(sorted(count), key=lambda s: count[s])
        on = sorted((p for p in reach if rows[g.net[p.number]][0] == side), key=lambda p: (p.dy, p.dx))
        seq = [rows[g.net[p.number]][1] for p in on]
        ahead, flipped = lis(seq), lis([-v for v in seq])
        flip = len(flipped) > len(ahead)
        pick = flipped if flip else ahead
        wired = tuple((on[i].number, rows[g.net[on[i].number]][2]) for i in pick)
        chosen = {on[i].number for i in pick}
        labelled = tuple((p.number, rows[g.net[p.number]][2]) for p in reach if p.number not in chosen)
        deltas = sorted(rows[g.net[on[i].number]][1] - (-on[i].dy if flip else on[i].dy) for i in pick)
        offset = deltas[(len(deltas) - 1) // 2] if deltas else 0
        plans.append(HeaderPlan(g.key, side, flip, wired, labelled, offset))  # type: ignore[arg-type]
    return plans


class HubRecipe:
    name = "hub"

    def claim(self, model: Model, block: BlockInfo, free: frozenset[str]) -> frozenset[str] | None:
        return None

    def draw(self, ctx, block: BlockInfo, members: frozenset[str]) -> list[Drawing]:
        raise LayoutFailure("hub: not available", block.id)


HUB = HubRecipe()
