"""Draw a perfboard layout guide (SVG) from a placed UHF breadboard radio IR - a documentation figure, not an artifact.

Usage: ``python docs/uhf_breadboard/make_perfboard_svg.py <ir.json> <out_dir>``

Also writes ``wiring.md``: per block, every net that touches its parts with
each pin as ``REF.pin(name)`` - the wiring list to solder from (the compiled
schematic connects its symbols by net labels, not drawn wires).

Each part sits where ``placement.rf_floorplan`` put it on the KiCad board,
with every pad moved to the nearest hole of a 2.54 mm perfboard grid (a
TO-92's 1.27 mm legs are spread to 2.54 mm: two pads never share a hole).
Each net is drawn as the shortest tree of straight wires between its holes:
red the supply rails, black the ground bus (a stub and a ground mark at each
hole: on a perfboard, a bare copper strip or the copper-clad back), blue the
RF nets, grey every other signal. The figure is a placement and wiring
guide, not a verified layout: the schematic (``<project>.kicad_sch``) is the
reference, and at 432 MHz the RF stages want the shortest leads over a
ground plane, not the wires drawn here. Only the KiCad library on disk is
read (the footprints' pads); nothing is written but the SVGs.
"""

from __future__ import annotations

import html
import json
import math
import sys
from pathlib import Path

from ai_eda.ir import CircuitIR, LibraryRef
from ai_eda.tools.kicad.geometry import pad_center
from ai_eda.tools.kicad.library import KicadLibrary

PITCH = 2.54
SCALE = 7.0  # px per mm
RAILS = {"V9", "V_TX", "V_RX", "V5T", "V5R", "VBAT", "VBAT_ON"}
RF_PREFIXES = ("TX_", "PA_", "LPF_", "ANT", "RX_IN", "RXQ", "SR_", "XB", "XE", "VAR")


def snap(v: float) -> float:
    return round(v / PITCH) * PITCH


def holes(ir: CircuitIR, lib: KicadLibrary) -> dict[tuple[str, str], tuple[float, float]]:
    """(ref, pad) -> the perfboard hole it goes into (distinct holes per part)."""
    out: dict[tuple[str, str], tuple[float, float]] = {}
    taken: set[tuple[float, float]] = set()
    by_ref = {c.ref: c for c in ir.components}
    for p in ir.pcb.placements:
        c = by_ref.get(p.component_ref)
        if c is None or c.footprint is None:
            continue
        fp = lib.load_footprint(lib.resolve_footprint(c.footprint))
        seen: set[str] = set()
        for pad in fp.pads:
            if not pad.number or pad.number in seen or pad.pad_type == "np_thru_hole":
                continue
            seen.add(pad.number)
            x, y = pad_center(p, pad)
            h = (snap(x), snap(y))
            step = 0
            while h in taken:  # spread pads that land on one hole (TO-92 1.27 mm, SOT-23)
                step += 1
                h = (snap(x) + PITCH * step, snap(y))
            taken.add(h)
            out[(c.ref, pad.number)] = h
    return out


def mst(points: list[tuple[float, float]]) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    pts = list(dict.fromkeys(points))
    if len(pts) < 2:
        return []
    inside = {pts[0]}
    edges = []
    while len(inside) < len(pts):
        best = min(((a, b) for a in inside for b in pts if b not in inside), key=lambda e: math.dist(*e))
        edges.append(best)
        inside.add(best[1])
    return edges


def draw(ir: CircuitIR, lib: KicadLibrary, title: str, box: tuple[float, float, float, float] | None) -> str:
    hole = holes(ir, lib)
    w, h = ir.pcb.outline.width_mm, ir.pcb.outline.height_mm  # type: ignore[union-attr]
    x0, y0, x1, y1 = box if box else (0.0, 0.0, w, h)
    W, H = (x1 - x0) * SCALE, (y1 - y0) * SCALE + 30
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" viewBox="0 0 {W:.0f} {H:.0f}" font-family="sans-serif">',
         f'<rect width="{W:.0f}" height="{H:.0f}" fill="#f4efe1"/>',
         f'<text x="8" y="20" font-size="14" fill="#333">{html.escape(title)}</text>']
    T = lambda x, y: ((x - x0) * SCALE, (y - y0) * SCALE + 30)  # noqa: E731
    gx = math.ceil(x0 / PITCH) * PITCH
    while gx <= x1:
        gy = math.ceil(y0 / PITCH) * PITCH
        while gy <= y1:
            px, py = T(gx, gy)
            s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.1" fill="#c9b98f"/>')
            gy += PITCH
        gx += PITCH
    inbox = lambda p: x0 <= p[0] <= x1 and y0 <= p[1] <= y1  # noqa: E731
    for n in ir.nets:
        pts = [hole[(pr.component_ref, pr.pin_number)] for pr in n.pins if (pr.component_ref, pr.pin_number) in hole]
        if n.name == "GND":
            for p in pts:
                if inbox(p):
                    px, py = T(*p)
                    s.append(f'<path d="M{px - 4:.1f},{py + 6:.1f}h8M{px - 2.5:.1f},{py + 8:.1f}h5M{px:.1f},{py:.1f}v6" stroke="#111" stroke-width="1.2" fill="none"/>')
            continue
        color = "#d22" if n.name in RAILS else "#1560c0" if n.name.startswith(RF_PREFIXES) else "#777"
        for a, b in mst(pts):
            if inbox(a) or inbox(b):
                (ax, ay), (bx, by) = T(*a), T(*b)
                s.append(f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" stroke="{color}" stroke-width="1.6" stroke-opacity="0.75"/>')
    parts: dict[str, list[tuple[float, float]]] = {}
    for (ref, _pad), p in hole.items():
        parts.setdefault(ref, []).append(p)
    vals = {c.ref: c.value or "" for c in ir.components}
    for ref, pts in parts.items():
        if not any(inbox(p) for p in pts):
            continue
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        (ax, ay), (bx, by) = T(min(xs) - 1.0, min(ys) - 1.0), T(max(xs) + 1.0, max(ys) + 1.0)
        s.append(f'<rect x="{ax:.1f}" y="{ay:.1f}" width="{bx - ax:.1f}" height="{by - ay:.1f}" rx="3" fill="#fff" fill-opacity="0.55" stroke="#555" stroke-width="0.8"/>')
        for p in pts:
            px, py = T(*p)
            s.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="#b87333" stroke="#5a3a1a" stroke-width="0.6"/>')
        s.append(f'<text x="{ax + 1:.1f}" y="{ay - 2:.1f}" font-size="8.5" fill="#000">{html.escape(ref)} {html.escape(vals.get(ref, ""))}</text>')
    s.append("</svg>")
    return "\n".join(s)


def wiring(ir: CircuitIR) -> str:
    """The wiring list: per RF block, each net touching its parts with every pin of that net (``REF.pin(name)``)."""
    by_ref = {c.ref: c for c in ir.components}
    lines = ["# 배선표 (블록별, 넷마다 연결할 핀)", "",
             "각 줄은 한 넷입니다: 그 핀들을 모두 한 점(또는 한 선)으로 납땜합니다. `GND`는 접지 버스, 다른 블록의 핀은 *기울임*으로 표시합니다. 전해 커패시터는 핀 1이 +입니다(설계 선택 cp_polarity: 부품의 극성 표시와 대조).", ""]
    for blk in ir.rf.blocks if ir.rf else []:  # type: ignore[union-attr]
        refs = set(blk.refs)
        lines += [f"## {blk.id} - {blk.title}", ""]
        for n in ir.nets:
            if not any(p.component_ref in refs for p in n.pins):
                continue
            if n.name == "GND":
                pins = [f"{p.component_ref}.{p.pin_number}" for p in n.pins if p.component_ref in refs]
                lines.append(f"- **GND** (접지 버스): {', '.join(pins)}")
                continue
            items = []
            for p in n.pins:
                c = by_ref[p.component_ref]
                pin = c.pin(p.pin_number)
                name = pin.name if pin is not None and pin.name and pin.name != p.pin_number else ""
                if not name and c.symbol is not None and c.symbol.name == "C_Polarized":
                    name = {"1": "+", "2": "-"}.get(p.pin_number, "")  # the template's cp_polarity choice: pin 1 = +
                elif not name and pin is not None and str(pin.electrical_type.value) == "output":
                    name = "OUT"
                text = f"{p.component_ref}.{p.pin_number}{'(' + name + ')' if name else ''}"
                items.append(text if p.component_ref in refs else f"*{text}*")
            lines.append(f"- **{n.name}**: {', '.join(items)}")
        lines.append("")
    lines += ["## 부품 값", "", "| 부품 | 값 | 설명 |", "|---|---|---|"]
    lines += [f"| {c.ref} | {c.value or ''} | {c.description} |" for c in ir.components]
    return "\n".join(lines) + "\n"


def main() -> None:
    ir = CircuitIR.model_validate_json(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    lib = KicadLibrary()
    (out / "perfboard_all.svg").write_text(draw(ir, lib, "만능기판 배치도 (전체, 2.54 mm 격자) - 빨강: 전원, 파랑: RF, 회색: 신호, 접지 기호: 접지 버스", None), encoding="utf-8")
    for blk in ir.rf.blocks if ir.rf else []:  # type: ignore[union-attr]
        if blk.region is None:
            continue
        r = blk.region
        box = (r.x.value, r.y.value, r.x.value + r.w.value, r.y.value + r.h.value)
        (out / f"perfboard_{blk.id}.svg").write_text(draw(ir, lib, f"만능기판 배치도: {blk.id} - {blk.title}", box), encoding="utf-8")
    (out / "wiring.md").write_text(wiring(ir), encoding="utf-8")
    print(json.dumps(sorted(p.name for p in out.glob("perfboard_*.svg"))))


if __name__ == "__main__":
    main()
