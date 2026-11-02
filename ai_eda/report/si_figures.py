"""The signal-integrity figures of the circuit report: Z0 against track width, per-net delay bars, the SPICE step responses.

Invariant: these are *views* under the rule of :mod:`ai_eda.report.figures`
(no status computed, nothing registered, saved or mutated; the same inputs
give byte-identical SVG). What each figure reads:

* :func:`z0_width_figure` - the stackup in ``ir.pcb.stackup`` (the
  geometry of each outer routing layer over its reference plane,
  :func:`~ai_eda.tools.calc.tline.line_geometry`) and the widths the IR
  copper of each net class carries; the curve is the registered
  ``calc.tline.microstrip`` (Hammerstad & Jensen) evaluated over a width
  sweep - a picture of the formula for this stack, like a template's theory
  curve, never a verdict (``si.impedance`` judges the routed segments). A
  board without a plane next to a routing layer has no curve: the function
  returns the reason instead (impedance is undefined without a reference
  plane).
* :func:`delay_bar_figure` - rows copied from the ``si.critical_length``
  result's ``details["nets"]`` (the routed delay, length and critical length
  the check recorded) and the check's own lists of electrically long nets;
  the only arithmetic is the threshold ``fraction x t_r`` the rule compares
  the delay with (delay > fraction x t_r is the same statement as
  length > l_crit).
* :func:`step_response_figure` - the far-end and near-end voltages of one
  ``spice.si.<net>`` transient read from its ngspice rawfile, which the
  caller locates through :func:`fresh_si_rawfile` (the latest result for the
  net, stamped with the current design hash, whose rawfile evidence is on
  disk with its recorded hash); the band and the recorded overshoot /
  undershoot are the result's own numbers.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_eda.ir import CircuitIR, ValidationResult, ValidationStatus
from ai_eda.report.figures import (
    AXIS,
    COLUMN_PX,
    GRID,
    INK,
    INK_SECONDARY,
    SERIES_COLOURS,
    Band,
    Figure,
    Marker,
    Series,
    _f,
    _g,
    _legend,
    _Scale,
    _svg_open,
    _text_width,
    _tick_labels,
    _title_text,
    esc,
    nice_ticks,
    svg_line_chart,
)
from ai_eda.tools.calc.tline import MICROSTRIP_U_RANGE, NO_REFERENCE_PLANE, NO_STACKUP, TLineRangeError, line_geometry, microstrip
from ai_eda.tools.spice.rawfile import parse as parse_rawfile

__all__ = [
    "DelayRow",
    "delay_bar_figure",
    "delay_rows",
    "fresh_si_rawfile",
    "step_response_figure",
    "z0_width_figure",
]

#: the routing layers the router uses (routing.maze's ``LAYERS``): each is a microstrip over its neighbouring plane
ROUTING_LAYERS: tuple[str, str] = ("F.Cu", "B.Cu")
#: points of the width sweep of the Z0 curve
Z0_SWEEP_POINTS = 120
#: the delay bar chart shows at most this many nets (the longest delays); the rest are counted in the caption
DELAY_BARS = 24
_ROW_H = 18
_BAR_H = 10
#: the step-response window after the rising edge, in units of (t_r + 2 t_d): the ringing of the lossless line has decayed by then
STEP_WINDOW_EDGES = 8.0


# --------------------------------------------------------------------------- Z0 against width


def _class_widths(ir: CircuitIR) -> dict[float, list[str]]:
    """Distinct track widths of the IR copper -> the net classes whose nets carry them (``ir.si`` order; no class: ``-``)."""
    pcb = ir.pcb
    si = ir.si
    out: dict[float, list[str]] = {}
    if pcb is None:
        return out
    for t in pcb.tracks:
        cls = si.class_of(t.net) if si is not None else None
        name = cls.name if cls is not None else "-"
        names = out.setdefault(float(t.width_mm), [])
        if name not in names:
            names.append(name)
    order = {c.name: i for i, c in enumerate(si.net_classes)} if si is not None else {}
    return {w: sorted(names, key=lambda n: (order.get(n, len(order)), n)) for w, names in sorted(out.items())}


def z0_width_figure(ir: CircuitIR, *, fig_id: str = "si_z0", width: int = COLUMN_PX) -> Figure | str:
    """Z0 of a microstrip on each outer routing layer against its width, with the routed widths marked (module docstring); the reason when there is no curve."""
    stackup = ir.pcb.stackup if ir.pcb is not None else None
    if stackup is None:
        return NO_STACKUP
    geoms: list[tuple[tuple[float, float, float], list[str], list[str]]] = []  # (h, t, er) -> layers, references
    reasons: list[str] = []
    for layer in ROUTING_LAYERS:
        g, why = line_geometry(stackup, layer)
        if g is None:
            reasons.append(f"{layer}: {why}")
            continue
        key = (float(g.h.value), float(g.t.value), float(g.er.value))
        for k, layers, refs in geoms:
            if k == key:
                layers.append(layer)
                refs.append(f"{g.reference_layer} {g.reference_net}")
                break
        else:
            geoms.append((key, [layer], [f"{g.reference_layer} {g.reference_net}"]))
    if not geoms:
        return "; ".join(reasons) or NO_REFERENCE_PLANE
    widths = _class_widths(ir)
    lo_w = min([0.05, *(w * 0.5 for w in widths)])
    hi_w = max([1.0, *(w * 1.5 for w in widths)])
    series: list[Series] = []
    for (h, t, er), layers, refs in geoms[:4]:
        w_lo = max(lo_w, MICROSTRIP_U_RANGE[0] * h, t / 1000.0 * 1.01)
        w_hi = min(hi_w, MICROSTRIP_U_RANGE[1] * h)
        xs: list[float] = []
        ys: list[float] = []
        for i in range(Z0_SWEEP_POINTS + 1):
            w = w_lo + (w_hi - w_lo) * i / Z0_SWEEP_POINTS
            try:
                z = microstrip(w, h, t, er).z0_ohm
            except (TLineRangeError, ValueError):
                continue
            xs.append(round(w, 9))
            ys.append(z)
        if len(xs) >= 2:
            series.append(Series(f"{' / '.join(layers)} (h {_g(h)} mm, t {_g(t)} µm, εr {_g(er)}; 기준 {' / '.join(refs)})", xs, ys))
    if not series:
        return "the microstrip formula refuses every width of the sweep for this stack"
    (h0, t0, er0), _layers, _refs = geoms[0]
    markers: list[Marker] = []
    for w, names in widths.items():
        try:
            z = microstrip(w, h0, t0, er0).z0_ohm
        except (TLineRangeError, ValueError):
            continue
        markers.append(Marker(w, z, f"{_g(w)} mm → {z:.1f} Ω ({', '.join(names)})"))
    bands: list[Band] = []
    if ir.si is not None:
        for c in ir.si.net_classes:
            if c.target_z0_ohm is None or c.z0_tol_rel is None:
                continue
            target, tol = float(c.target_z0_ohm.value), float(c.z0_tol_rel.value)
            bands.append(Band("y", target * (1 - tol), target * (1 + tol), f"{c.name} 목표 {_g(target)} Ω ± {tol:.0%}"))
    title = f"{ir.project.id}: 마이크로스트립 Z0 대 트랙 폭"
    svg = svg_line_chart(series, title=title, x_label="트랙 폭 (mm)", y_label="Z0 (Ω)", bands=bands[:2], markers=markers, width=width, height=420, clip_id=fig_id)
    marked = "; ".join(m.label for m in markers) or "배선된 트랙 없음"
    caption = (
        "calc.tline.microstrip (Hammerstad & Jensen, 두께 보정 포함, 준정적·무손실)을 이 적층의 외층 기하에 넣어 폭만 바꿔 그린 곡선입니다. "
        f"점은 IR 의 트랙이 실제로 가진 폭(그 폭을 쓰는 넷 클래스): {marked}. "
        + ("띠는 넷 클래스의 목표 임피던스 허용 범위입니다. " if bands else "")
        + "솔더 마스크·분산·손실은 모델에 없고, 실제 임피던스는 fab 의 측정만이 말합니다. 그림은 판정이 아니며 판정은 si.impedance 결과입니다."
        + (f" (그리지 않은 층: {'; '.join(reasons)})" if reasons else "")
    )
    return Figure(fig_id, title, caption, svg)


# --------------------------------------------------------------------------- delay bars


@dataclass(frozen=True)
class DelayRow:
    """One routed net of the ``si.critical_length`` record: its delay, length, critical length and the rule's threshold ``fraction x t_r``."""

    net: str
    net_class: str
    delay_s: float
    length_mm: float
    l_crit_mm: float | None
    threshold_s: float | None
    long: bool
    bound: bool
    #: the record's "possibly long": the whole copper (an upper bound of every path) is over l_crit, no pad-to-pad path extracted
    possibly: bool = False


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def delay_rows(result: ValidationResult) -> list[DelayRow]:
    """The routed nets of a recorded ``si.critical_length`` result (``details["nets"]`` with a delay), copied; ``long`` from the result's own lists."""
    details = result.details if isinstance(result.details, dict) else {}
    long_nets = {str(n) for key in ("long_over_plane", "long_without_plane") for n in details.get(key) or []}
    possibly = {str(n) for n in details.get("possibly_long") or []}
    out: list[DelayRow] = []
    for row in details.get("nets") or []:
        if not isinstance(row, dict):
            continue
        delay_ps, length = _num(row.get("delay_ps")), _num(row.get("length_mm"))
        if delay_ps is None or length is None:
            continue
        t_r, fraction = _num(row.get("t_rise_s")), _num(row.get("fraction"))
        out.append(DelayRow(
            net=str(row.get("net")), net_class=str(row.get("class") or "-"), delay_s=delay_ps * 1e-12, length_mm=length,
            l_crit_mm=_num(row.get("l_crit_mm")), threshold_s=t_r * fraction if t_r is not None and fraction is not None else None,
            long=str(row.get("net")) in long_nets, bound=bool(row.get("bound")), possibly=str(row.get("net")) in possibly,
        ))
    return out


def delay_bar_figure(rows: Sequence[DelayRow], *, title: str, fig_id: str = "si_delay", limit: int = DELAY_BARS, width: int = COLUMN_PX) -> Figure:
    """Horizontal bars of the ``limit`` longest routed delays against the critical-length threshold (module docstring); ``ValueError`` without rows."""
    if not rows:
        raise ValueError("no routed net with a recorded delay")
    ranked = sorted(rows, key=lambda r: (-r.delay_s, r.net))
    shown = ranked[:limit]
    thresholds = sorted({r.threshold_s for r in shown if r.threshold_s is not None})
    x_hi = max([r.delay_s for r in shown] + thresholds) * 1.08
    ticks = nice_ticks(0.0, x_hi)
    x_hi = max(x_hi, ticks[-1])
    labels = [f"{r.net} ({r.net_class})" for r in shown]
    values = [f"{r.delay_s * 1e12:.1f} ps · {r.length_mm:.1f} mm" + (f" / l_crit {r.l_crit_mm:.1f}" if r.l_crit_mm is not None else "") for r in shown]
    ml = int(round(12 + max(_text_width(t, 11) for t in labels) + 8))
    mr = int(round(max(_text_width(t, 11) for t in values) + 16))
    top = 44
    legend_entries = [("box", SERIES_COLOURS[0], "짧음 (기록)"), ("box", SERIES_COLOURS[1], "김: l_crit 초과 (기록)")]
    if any(r.possibly for r in shown):
        legend_entries.append(("box", SERIES_COLOURS[3], "길 수도 있음: 경로 미추출, 전체 구리 (기록)"))
    if len(thresholds) >= 1:
        legend_entries.append(("line", INK, "f × t_r (넘으면 길이 > l_crit)"))
    legend, top = _legend(legend_entries, 12, 46, width - 12)
    top += 4
    height = top + _ROW_H * len(shown) + 50
    out = _svg_open(width, height, title)
    out.append(_title_text(title, 12, 24))
    out += legend
    bottom = top + _ROW_H * len(shown)
    sx = _Scale(0.0, x_hi, ml, width - mr, False)
    out.append('<g class="grid">')
    for t in ticks:
        out.append(f'<line x1="{_f(sx(t))}" y1="{_f(top)}" x2="{_f(sx(t))}" y2="{_f(bottom)}" stroke="{GRID}" stroke-width="1"/>')
    out.append("</g>")
    out.append(f'<line class="axis" x1="{_f(ml)}" y1="{_f(top)}" x2="{_f(ml)}" y2="{_f(bottom)}" stroke="{AXIS}" stroke-width="1"/>')
    out.append(f'<line class="axis" x1="{_f(ml)}" y1="{_f(bottom)}" x2="{_f(width - mr)}" y2="{_f(bottom)}" stroke="{AXIS}" stroke-width="1"/>')
    out.append('<g class="ticks">')
    for t, text in zip(ticks, _tick_labels(ticks, "s", False)):
        out.append(f'<text x="{_f(sx(t))}" y="{_f(bottom + 16)}" text-anchor="middle" fill="{INK_SECONDARY}">{esc(text)}</text>')
    out.append("</g>")
    out.append('<g class="bars">')
    for i, (r, label, value) in enumerate(zip(shown, labels, values)):
        yc = top + _ROW_H * (i + 0.5)
        x1 = sx(r.delay_s)
        colour = SERIES_COLOURS[1] if r.long else (SERIES_COLOURS[3] if r.possibly else SERIES_COLOURS[0])
        rr = min(3.0, _BAR_H / 2, max(0.0, x1 - ml))
        y0 = yc - _BAR_H / 2
        path = (f"M{_f(ml)},{_f(y0)} H{_f(x1 - rr)} Q{_f(x1)},{_f(y0)} {_f(x1)},{_f(y0 + rr)} V{_f(y0 + _BAR_H - rr)} "
                f"Q{_f(x1)},{_f(y0 + _BAR_H)} {_f(x1 - rr)},{_f(y0 + _BAR_H)} H{_f(ml)} Z")
        out.append(f'<path class="bar" data-net="{esc(r.net)}" data-value="{_g(r.delay_s * 1e12)}" data-long="{"true" if r.long else "false"}" d="{path}" fill="{colour}"/>')
        out.append(f'<text class="category" x="{_f(ml - 6)}" y="{_f(yc + 4)}" text-anchor="end" font-size="11" fill="{INK_SECONDARY}">{esc(label)}</text>')
        out.append(f'<text class="bar-label" x="{_f(width - mr + 8)}" y="{_f(yc + 4)}" font-size="11">{esc(value)}</text>')
        if r.threshold_s is not None and len(thresholds) > 1:
            xt = sx(r.threshold_s)
            out.append(f'<line class="threshold" x1="{_f(xt)}" y1="{_f(yc - _ROW_H / 2 + 1)}" x2="{_f(xt)}" y2="{_f(yc + _ROW_H / 2 - 1)}" stroke="{INK}" stroke-width="1.5"/>')
    out.append("</g>")
    if len(thresholds) == 1:
        xt = sx(thresholds[0])
        out.append(f'<line class="threshold" x1="{_f(xt)}" y1="{_f(top - 2)}" x2="{_f(xt)}" y2="{_f(bottom)}" stroke="{INK}" stroke-width="1.5"/>')
    out.append(f'<text class="axis-title" x="{_f((ml + width - mr) / 2)}" y="{_f(height - 12)}" text-anchor="middle">배선 지연 (s)</text>')
    out.append("</svg>")
    rest = len(ranked) - len(shown)
    bound = any(r.bound for r in shown)
    threshold_text = ", ".join(f"{t * 1e12:g} ps" for t in thresholds) or "기록 없음"
    caption = (
        f"si.critical_length 결과가 기록한 넷별 선로 지연(가장 긴 패드-패드 경로, 또는 패드가 둘인 넷의 구리; 트랙 + 비아 배럴) 중 긴 {len(shown)}개"
        + (f" (나머지 {rest}개는 모두 이보다 짧음)" if rest > 0 else "")
        + f". 세로선은 규칙의 문턱 f × t_r = {threshold_text}: 지연이 이 선을 넘는 것은 배선 길이가 l_crit = f × t_r / t_pd 보다 긴 것과 같은 말입니다. "
        "막대 색은 검사 결과의 '긴 넷' 목록을 옮긴 것이며, 막대 끝의 수는 지연 · 길이 / l_crit 입니다."
        + (" '길 수도 있음' 막대는 경로를 뽑지 못해 넷 전체 구리(어떤 경로보다도 긴 상한)를 그린 것입니다." if any(r.possibly for r in shown) else "")
        + (" 기준 평면이 없는 층의 지연은 상한 sqrt(εr)/c0 로 계산한 값입니다 (실제 지연은 더 짧음)." if bound else "")
        + " 그림은 판정을 새로 하지 않습니다."
    )
    return Figure(fig_id, title, caption, "\n".join(out) + "\n")


# --------------------------------------------------------------------------- SPICE step responses


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def fresh_si_rawfile(ir: CircuitIR, result: ValidationResult) -> Path | str:
    """The rawfile of a ``spice.si.<net>`` result when it is evidence about the current IR, else why not.

    The result must be a tool-backed PASS / FAIL stamped with the current
    design hash, and its rawfile evidence must be on disk with the recorded
    content hash (a rawfile changed or removed since is no evidence).
    """
    if result.status not in (ValidationStatus.PASS, ValidationStatus.FAIL) or not result.is_tool_backed:
        return f"{result.check_id} 은 {result.status} 이라 파형이 없습니다"
    if result.ir_hash != ir.content_hash():
        return f"{result.check_id} 은 이전 IR 버전의 결과입니다"
    for ev in result.evidence:
        if ev.path and ev.path.endswith(".raw"):
            path = Path(ev.path)
            if not path.is_file():
                return f"{result.check_id} 의 rawfile {path.name} 이 디스크에 없습니다"
            if ev.content_hash is None or _sha256(path) != ev.content_hash:
                return f"{result.check_id} 의 rawfile {path.name} 이 기록된 해시와 다릅니다"
            return path
    return f"{result.check_id} 에 rawfile 증거가 없습니다"


def step_response_figure(result: ValidationResult, rawfile: Path, *, fig_id: str, width: int = COLUMN_PX) -> Figure:
    """The far-end ``v(b)`` and near-end ``v(a)`` of a ``spice.si.<net>`` transient over the rising edge (module docstring); ``ValueError`` when the rawfile lacks them."""
    details = result.details if isinstance(result.details, dict) else {}
    net = str(details.get("net") or result.check_id.rsplit(".", 1)[-1])
    plot = parse_rawfile(rawfile).as_plot_vectors()
    time, vb, va = plot.get("time"), plot.get("b"), plot.get("a")
    if not time or not vb or not va or not (len(time) == len(vb) == len(va)):
        raise ValueError(f"rawfile {rawfile.name} has no time / v(a) / v(b) vectors of one length")
    t_r = _num(details.get("t_rise_s")) or 0.0
    td = _num(details.get("td_s")) or 0.0
    hold = _num(details.get("hold_s"))
    window = STEP_WINDOW_EDGES * (t_r + 2.0 * td)
    if hold is not None:
        window = min(window, hold)
    if not window > 0:
        window = time[-1]
    keep = [i for i, t in enumerate(time) if t <= window]
    xs = [time[i] for i in keep]
    series = [Series("먼 끝 v(B) (부하 쪽)", xs, [vb[i] for i in keep]), Series("가까운 끝 v(A) (드라이버 쪽)", xs, [va[i] for i in keep])]
    tol = _num(details.get("band_rel"))
    bands = [Band("y", 1.0 - tol, 1.0 + tol, f"±{tol:.0%} 대역")] if tol is not None else []
    measured = details.get("measured") if isinstance(details.get("measured"), dict) else {}
    markers: list[Marker] = []
    if keep:
        i_max = max(keep, key=lambda i: (vb[i], -i))
        over = _num(measured.get("overshoot_rel"))
        markers.append(Marker(time[i_max], vb[i_max], f"최대 {vb[i_max]:.3f} V" + (f" (오버슈트 {over:.1%})" if over is not None else "")))
    title = f"{result.check_id}: 1 V 계단 응답 (상승 에지, {result.status})"
    svg = svg_line_chart(series, title=title, x_label="시간 (s)", y_label="전압 (V)", bands=bands, markers=markers, width=width, height=400, clip_id=fig_id)
    z0, length = _num(details.get("z0_ohm")), _num(details.get("length_mm"))
    r_drive, c_load = _num(details.get("r_drive_ohm")), _num(details.get("c_load_f"))
    under = _num(measured.get("undershoot_rel"))
    caption = (
        f"{net}: ngspice 무손실 전송선 T(Z0 {z0:.2f} Ω, t_d {td * 1e12:.1f} ps; 배선 {length:.2f} mm)를 R_s {_g(r_drive or 0)} Ω, t_r {t_r * 1e9:g} ns 의 0→1 V 펄스로 구동하고 "
        f"먼 끝에 {_g((c_load or 0) * 1e12)} pF 를 단 결과의 rawfile({rawfile.name})을 그대로 그린 것입니다. 창은 상승 에지 뒤 {window * 1e9:.2f} ns "
        "(선형 회로라 하강 에지는 거울상). "
        + (f"기록된 판정 값: 오버슈트 {_num(measured.get('overshoot_rel')):.1%}, 언더슈트 {under:.1%}, 대역 ±{tol:.0%} → {result.status}. " if under is not None and tol is not None and _num(measured.get('overshoot_rel')) is not None else "")
        + "판정은 기록된 spice.si 결과의 것이며 그림은 새로 판정하지 않습니다."
    ) if z0 is not None and length is not None else f"{net}: rawfile {rawfile.name} 의 v(B), v(A). 판정은 기록된 spice.si 결과의 것입니다."
    return Figure(fig_id, title, caption, svg)
