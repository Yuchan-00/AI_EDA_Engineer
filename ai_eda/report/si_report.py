"""The signal-integrity text of the Korean stage reports: the circuit report's "임피던스·타이밍" section and the theory report's transmission-line section.

Invariant: a *view* under the rule of :mod:`ai_eda.report.stages` - every
status printed is a stored ``ValidationResult.status`` copied verbatim, no
verdict is computed, nothing is registered, hashed or saved, no wall-clock
and no absolute path (evidence files by basename, stored messages through
:func:`~ai_eda.report.stages.strip_paths`). The numbers are the IR's own
(``ir.si`` net classes and timing paths with their provenance,
``ir.pcb.stackup``, the routed copper) and the recorded ``si.*`` /
``spice.si.*`` results' details; where a formula is shown "with the IR's
numbers", the value is the registered calculator
(:mod:`ai_eda.tools.calc.tline`) evaluated on those numbers - a display
value named as such, like the circuit report's IPC-2221 reference values,
never written to the IR and never a verdict. A design without ``ir.si`` gets
one sentence saying so.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

from ai_eda.design.base import NO_RECORD, quantity
from ai_eda.ir import CircuitIR, NetClass, Provenance, ProvenanceKind, Stackup, Traced, ValidationResult, ValidationStatus
from ai_eda.tools.calc.tline import C0, NO_REFERENCE_PLANE, TLineRangeError, critical_length_mm, line_geometry, microstrip, propagation_delay

if TYPE_CHECKING:
    from ai_eda.report.stages import ReportFigures

#: the figure slots of the circuit report's SI section (filled by :func:`ai_eda.report.stages.si_figures_of`)
SLOT_SI_BOARD = "si_board"
SLOT_SI_Z0 = "si_z0"
SLOT_SI_DELAY = "si_delay"
SLOT_SI_STEP = "si_step"
#: what the SI section / theory section says for a design without ``ir.si``
NO_SI = "이 설계에는 신호 무결성 제약(`ir.si`: 넷 클래스·타이밍 경로)이 없어 임피던스·타이밍 검사를 하지 않았습니다"
#: the routing layers (routing.maze's ``LAYERS``)
ROUTING_LAYERS: tuple[str, str] = ("F.Cu", "B.Cu")
#: the check-id prefixes the SI section copies
SI_PREFIXES: tuple[str, ...] = ("si.", "spice.si.")
_CRITICAL = "si.critical_length"
_STATUS_ORDER = {s: i for i, s in enumerate((ValidationStatus.FAIL, ValidationStatus.NOT_VERIFIED, ValidationStatus.PASS, ValidationStatus.NOT_APPLICABLE))}


def _st():
    """The stage-report helpers (imported lazily: :mod:`ai_eda.report.stages` imports this module)."""
    from ai_eda.report import stages

    return stages


def _mm(value: float | None, digits: int = 3) -> str:
    return NO_RECORD if value is None else f"{value:.{digits}f} mm"


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _tv(t: Traced | None, unit: str | None = None) -> str:
    """A traced SI number for a table cell: ``quantity`` for an SI unit, plain for mm / a ratio (a ratio in %)."""
    if t is None:
        return "-"
    v = float(t.value)
    u = unit if unit is not None else t.unit
    if u is None:
        return f"{v * 100:g} %" if 0 < v <= 1 else f"{v:g}"
    if u == "mm":
        return f"{v:.6g} mm"
    if u == "degC":
        return f"{v:g} °C"
    return quantity(v, u, 4)


def _prov_origin(p: Provenance) -> str:
    """How a reader should weigh a provenance (the stage reports' words; a confirmed template choice as such)."""
    st = _st()
    if p.kind is ProvenanceKind.USER_REQUIREMENT and (p.note or "").startswith(st.CHOICE_NOTE_PREFIX):
        return st.CHOICE_LABEL
    return st._provenance_text(p)


def _origin(t: Traced) -> str:
    """How a reader should weigh a traced SI number (:func:`_prov_origin` of its provenance)."""
    return _prov_origin(t.provenance)


def _results(ir: CircuitIR) -> list[ValidationResult]:
    """The latest ``si.*`` / ``spice.si.*`` results, in check-id order (the critical-length rule first)."""
    latest = ir.validation.latest_by_check()
    rows = [r for k, r in latest.items() if k.startswith(SI_PREFIXES)]
    return sorted(rows, key=lambda r: (r.check_id != _CRITICAL, not r.check_id.startswith("si."), r.check_id))


#: a stored message longer than this is cut in the SI section (the rows below it and the result itself carry the rest)
MESSAGE_CHARS = 600


def _tool(r: ValidationResult) -> str:
    st = _st()
    return f"({st._tool_text(r.tool, r.tool_version)})" if r.tool else "(도구 없음)"


def _message(text: str) -> str:
    st = _st()
    cell = st._cell(text)
    return cell if len(cell) <= MESSAGE_CHARS else cell[:MESSAGE_CHARS].rstrip() + " … (이하 생략: 아래 표와 검사 결과의 기록에 전부 있음)"


def _result_line(r: ValidationResult) -> str:
    return f"- `{r.check_id}`: **{r.status}** {_tool(r)} — {_message(r.message)}"


def router_version_text(ir: CircuitIR) -> str:
    """What the routed copper's provenance says about routing.maze 0.2 identity (copied from the record; a board-level fact, never per net)."""
    from ai_eda.tools.routing.maze import ROUTER_ID, ROUTER_RULES_VERSION, ROUTER_VERSION

    tracks = [t for t in (ir.pcb.tracks if ir.pcb is not None else []) if t.provenance.tool == ROUTER_ID]
    if not tracks:
        return "이 보드에는 라우터가 배선한 트랙이 없습니다."
    versions = {t.provenance.tool_version for t in tracks}
    ruled = sorted({t.net for t in tracks if any(e.startswith("rule:") for e in t.provenance.derived_from)})
    if versions == {ROUTER_VERSION}:
        return (f"이 보드는 어떤 넷에도 라우터 규칙이 없어 routing.maze {ROUTER_VERSION} 와 바이트 단위로 같게 배선되었습니다(모든 트랙의 provenance 가 {ROUTER_VERSION}). "
                f"0.2 와의 바이트 동일성은 이처럼 보드 전체에 규칙이 하나도 없을 때만 성립합니다.")
    return (f"이 보드는 {len(ruled)} 개 넷({', '.join(f'`{n}`' for n in ruled[:12])}{' …' if len(ruled) > 12 else ''})에 라우터 규칙이 있어, 모든 넷이 한 협상에서 "
            f"배선되었고 모든 트랙이 routing.maze {ROUTER_RULES_VERSION} 으로 기록되어 있습니다. 규칙이 없는 넷도 보드의 기본 폭·간격을 쓰지만 협상을 함께 하므로 "
            f"0.2 와 다른 경로일 수 있습니다: 0.2 와의 바이트 동일성은 어떤 넷에도 규칙이 없는 보드에서만 성립하며, 이 보드는 그렇지 않습니다.")


# --------------------------------------------------------------------------- the stackup


def stackup_lines(stack: Stackup | None) -> list[str]:
    """The stackup as a table (copper and dielectrics top to bottom, with each number's origin), or why there is none."""
    st = _st()
    if stack is None:
        return [f"적층 {NO_RECORD}: `ir.pcb.stackup` 이 없으므로 모든 임피던스·지연 수치는 NOT_VERIFIED 'no stackup' 입니다.", ""]
    rows: list[list[object]] = []
    for i, c in enumerate(stack.copper):
        plane = f"평면: `{c.plane_net.value}`" if c.plane_net is not None else "신호"
        rows.append([c.name, "구리", f"{float(c.thickness_um.value):g} µm", "-", plane, _origin(c.thickness_um)])
        if i < len(stack.dielectrics):
            d = stack.dielectrics[i]
            freq = f" @ {quantity(float(d.er_frequency_hz.value), 'Hz', 4)}" if d.er_frequency_hz is not None else " (주파수 기록 없음)"
            rows.append([f"({c.name} / {stack.copper[i + 1].name})", f"유전체 ({d.kind.value})", f"{float(d.thickness_mm.value):g} mm", f"{float(d.er.value):g}{freq}", "-", _origin(d.er)])
    out = [st._table(["층", "종류", "두께", "εr", "역할", "출처"], rows), ""]
    out.append(f"- 적층 두께 합 (바깥 구리 면 사이): {stack.board_thickness_mm():.4g} mm, 구리 {stack.layer_count}층, 평면 층 "
               + (", ".join(f"{c.name} = `{c.plane_net.value}`" for c in stack.plane_layers()) or "없음"))
    note = stack.provenance.note or ""
    if note:
        out.append(f"- 적층 전체의 출처: {_prov_origin(stack.provenance)} — {st._cell(note)}")
    out.append("")
    return out


# --------------------------------------------------------------------------- net classes and timing paths


def _class_origins(c: NetClass) -> str:
    from ai_eda.ir.si import CLASS_UNITS

    seen: list[str] = []
    for field in CLASS_UNITS:
        t = getattr(c, field)
        if t is not None:
            o = _origin(t)
            if o not in seen:
                seen.append(o)
    return ", ".join(seen) or "-"


def class_lines(ir: CircuitIR) -> list[str]:
    st = _st()
    si = ir.si
    assert si is not None
    members: dict[str, list[str]] = {c.name: [] for c in si.net_classes}
    for n in ir.nets:
        c = si.class_of(n.name)
        if c is not None:
            members[c.name].append(n.name)
    rows: list[list[object]] = []
    for c in si.net_classes:
        declared = ", ".join(f"`{n}`" for n in c.nets) or ("(선언 없음: 다른 클래스에 없는 모든 넷)" if c.default else "-")
        promoted = ", ".join(f"`{p.net}`" for p in c.promoted) or "-"
        target = f"{_tv(c.target_z0_ohm)} ± {_tv(c.z0_tol_rel)}" if c.target_z0_ohm is not None else "-"
        if c.target_zdiff_ohm is not None:
            target += f"; 차동 {_tv(c.target_zdiff_ohm)} ± {_tv(c.zdiff_tol_rel)} ({', '.join(p.name for p in c.pairs)})"
        budget = "; ".join(x for x in (
            f"길이 ≤ {_tv(c.max_length_mm)}" if c.max_length_mm is not None else "",
            f"지연 ≤ {_tv(c.max_delay_s)}" if c.max_delay_s is not None else "",
            f"그룹 {c.match_group} 스큐 ≤ {_tv(c.max_skew_s)}" if c.match_group is not None else "",
        ) if x) or "-"
        rows.append([f"`{c.name}`" + (" (기본)" if c.default else ""), declared, promoted, len(members[c.name]), target,
                     _tv(c.min_width_mm) if c.min_width_mm is not None else "-", budget, f"`{c.promote_to}`" if c.promote_to else "-"])
    out = [st._table(["클래스", "선언된 넷", "승격된 넷", "배선하는 넷 수", "목표 임피던스", "최소 폭", "길이·지연 예산", "승격 대상"], rows), ""]
    drivers: dict[str, list[str]] = {}
    origins: dict[str, list[str]] = {}
    for c in si.net_classes:
        if c.t_rise_s is not None:
            text = f"t_r {_tv(c.t_rise_s)}, R_s {_tv(c.r_drive_ohm)}, C_L {_tv(c.c_load_f)}, 대역 ±{_tv(c.ringing_tol_rel)}"
            if c.driver:
                text += f" (`{c.driver}` 의 데이터시트 t_rise / r_out / c_in 이 grounding 되면 `{c.driver}` 가 구동하는(출력·양방향 핀이 있는) 넷에서만 그 값으로 대체)"
            else:
                text += " (드라이버를 지정하지 않음: 이 선택값이 그대로 쓰임)"
            drivers.setdefault(text, []).append(c.name)
        origins.setdefault(_class_origins(c), []).append(c.name)
    for text, names in drivers.items():
        out.append(f"- 드라이버 모델 (임계 길이 규칙과 SPICE 검사) — {', '.join(f'`{n}`' for n in names)}: {text}")
    for text, names in origins.items():
        out.append(f"- 수치의 출처 — {', '.join(f'`{n}`' for n in names)}: {text}")
    if si.critical_fraction is not None:
        out.append(f"- 임계 길이 규칙의 비율 f = {float(si.critical_fraction.value):g} ({_origin(si.critical_fraction)}): l_crit = f × t_r / t_pd.")
    if si.timing_paths:
        out += ["", "타이밍 경로 (동기 인터페이스):", ""]
        trows: list[list[object]] = []
        for p in si.timing_paths:
            terms = []
            for term, label in (("t_co_max_s", "t_co,max"), ("t_co_min_s", "t_co,min"), ("t_su_min_s", "t_su,min"), ("t_h_min_s", "t_h,min")):
                own = getattr(p, term)
                src = p.terms_from.get(term)
                terms.append(f"{label} = {_tv(own)}" if own is not None else f"{label} = {NO_RECORD}" + (f" ← `{src}`" if src else ""))
            trows.append([f"`{p.name}`", f"`{p.clock_net}` → " + ", ".join(f"`{n}`" for n in p.data_nets), p.direction or "-",
                          f"{_tv(p.f_clk_hz)} ({_origin(p.f_clk_hz)})" if p.f_clk_hz is not None else NO_RECORD,
                          f"{float(p.capture_fraction.value):g} × T" if p.capture_fraction is not None else NO_RECORD, "; ".join(terms)])
        out += [st._table(["경로", "클럭 → 데이터", "방향", "f_clk", "발사→캡처", "데이터시트 항"], trows)]
    out.append("")
    return out


# --------------------------------------------------------------------------- formulas with the IR's numbers


def _widths(ir: CircuitIR) -> dict[float, list[str]]:
    from ai_eda.report.si_figures import _class_widths

    return _class_widths(ir)


def formula_lines(ir: CircuitIR) -> list[str]:
    """The line model per routing layer with the stackup's numbers, the width for each impedance target (copied from ``si.impedance``), Z0 / e_eff / t_pd / l_crit per routed width."""
    si = ir.si
    assert si is not None
    stack = ir.pcb.stackup if ir.pcb is not None else None
    out = ["### 식 (IR 의 수치를 넣은 것)", ""]
    if stack is None:
        return out + [f"적층 {NO_RECORD}: 선로 모델의 입력(h, t, εr)이 없어 식에 넣을 수가 없습니다.", ""]
    fraction = float(si.critical_fraction.value) if si.critical_fraction is not None else None
    t_rs = sorted({float(c.t_rise_s.value) for c in si.net_classes if c.t_rise_s is not None})
    t_r = t_rs[0] if len(t_rs) == 1 else None
    widths = _widths(ir)
    out += [
        "계산은 모두 등록된 계산기 `calc.tline.*` 의 준정적(TEM, 무손실) 식입니다. 아래의 Z0 · e_eff · t_pd · l_crit 은 IR 의 적층과 트랙 폭을 같은 계산기에 넣은 **표시값**이며 "
        "IR 에 쓰지 않고 판정도 아닙니다(판정은 아래의 기록된 `si.*` 결과).", "",
    ]
    geoms: list[tuple[tuple[float, float, float], list[Any]]] = []
    bounds: dict[float, list[tuple[str, str]]] = {}
    general = False
    for layer in ROUTING_LAYERS:
        g, why = line_geometry(stack, layer)
        if g is not None:
            key = (float(g.h.value), float(g.t.value), float(g.er.value))
            for k, gs in geoms:
                if k == key:
                    gs.append(g)
                    break
            else:
                geoms.append((key, [g]))
            continue
        reason = why or NO_REFERENCE_PLANE
        d = stack.dielectrics[0 if layer == "F.Cu" else len(stack.dielectrics) - 1]
        bounds.setdefault(float(d.er.value), []).append((layer, reason))
    for er, layers in bounds.items():
        t_pd = propagation_delay(er)
        reasons = "; ".join(f"`{reason}`" for _layer, reason in layers)
        out += [f"- **{' / '.join(layer for layer, _r in layers)}**: 기준 평면이 없습니다 ({reasons}). 임피던스는 정의되지 않으므로 계산하지 않습니다. "
                "지연은 상한만 있습니다 — 전기장이 유전체와 공기에 나뉘므로 e_eff ≤ εr:", "",
                f"      t_pd ≤ √εr / c0 = √{er:g} / {C0:.0f} m/s = {t_pd * 1e9:.4f} ps/mm   (`calc.tline.tpd`, εr = 인접 유전체)"]
        if fraction is not None and t_r is not None:
            out.append(f"      l_crit ≥ f × t_r / t_pd = {fraction:g} × {quantity(t_r, 's')} / {t_pd * 1e9:.4f} ps/mm = {critical_length_mm(t_r, t_pd, fraction):.2f} mm   (`calc.tline.critical_length`)")
        out += ["", "  상한 t_pd 로 구한 l_crit 은 가장 짧은 값이라 넷을 '길다' 고 보는 쪽(보수적)입니다.", ""]
    for (h, t_um, er), gs in geoms:
        if not general:
            general = True
            out += [
                "- **마이크로스트립** (기준 평면 위의 외층 트랙; Hammerstad & Jensen, 두께 보정 포함 — 이론 보고서에 식 전체):", "",
                "      Z0 = Z01(u_r)/√e_eff(u_r),  u = w/h (u_r: 두께 보정 폭)",
                "      t_pd = √e_eff / c0,  l_crit = f × t_r / t_pd", "",
            ]
        refs = " / ".join(f"{g.layer} 는 {g.reference_layer} ({g.reference_net} 평면)" for g in gs)
        g = gs[0]
        out += [f"- **{' / '.join(x.layer for x in gs)}** — 기준 평면: {refs}; h = {h:g} mm (`{g.h_id}`), t = {t_um:g} µm (`{g.t_id}`), εr = {er:g} (`{g.er_id}`)"
                + (" — 층마다 같은 수치" if len(gs) > 1 else "") + ".", ""]
        for w, names in widths.items():
            try:
                r = microstrip(w, h, t_um, er)
            except (TLineRangeError, ValueError) as e:
                out.append(f"      w = {w:g} mm ({', '.join(names)}): 계산기가 거부 ({e})")
                continue
            t_pd = propagation_delay(r.e_eff)
            out.append(f"      w = {w:g} mm ({', '.join(names)}): u = {w / h:.4g}, Z0 = {r.z0_ohm:.2f} Ω, e_eff = {r.e_eff:.4f}")
            line = f"        t_pd = √{r.e_eff:.4f} / c0 = {t_pd * 1e9:.4f} ps/mm"
            if fraction is not None and t_r is not None:
                line += f",  l_crit = {fraction:g} × {quantity(t_r, 's')} / t_pd = {critical_length_mm(t_r, t_pd, fraction):.2f} mm"
            out.append(line)
        out.append("")
    via = stack.span_mm("F.Cu", "B.Cu")
    er_max = max(float(d.er.value) for d in stack.dielectrics)
    out += [f"- **비아**: 관통 비아 배럴 길이 = F.Cu..B.Cu = {via:.4g} mm, 지연 상한 √εr,max / c0 = √{er_max:g} / c0 = {propagation_delay(er_max) * 1e9:.4f} ps/mm "
            "(비아 하나마다 넷의 길이와 지연에 더함).", ""]
    for c in si.net_classes:
        r = ir.validation.latest(f"si.impedance.{c.name}")
        if c.target_z0_ohm is None or r is None or not isinstance(r.details, dict) or r.details.get("width_derivation") is None:
            continue
        w = r.details.get("controlled_width_mm")
        if w is None:
            out += [f"- **{c.name} 의 폭**: 제어 폭 없음 — `si.impedance.{c.name}` 기록: {_st()._cell(r.details['width_derivation'])}", ""]
            continue
        out += [f"- **{c.name} 의 폭** (`calc.tline.width_for_z0.microstrip`: 순방향 식을 이분법으로 풀고 0.01 mm 단위로 올림; `si.impedance.{c.name}` 기록 그대로):", "",
                f"      {_st()._cell(r.details['width_derivation'])}", f"      → 배선 폭 {w} mm", ""]
    for c in si.net_classes:
        if c.min_width_mm is None:
            continue
        p = c.min_width_mm.provenance
        out += [f"- **{c.name} 의 최소 폭** ({_origin(c.min_width_mm)}): w_min = {float(c.min_width_mm.value):.6g} mm"
                + (f" — 식: {_st()._cell(p.note)}" if p.note else "")
                + (f"; I = {_tv(c.power_current_a)}, ΔT = {_tv(c.power_temp_rise_c)}" if c.power_current_a is not None and c.power_temp_rise_c is not None else "")
                + f"; 입력: {', '.join(f'`{x}`' for x in p.derived_from) or NO_RECORD}.", ""]
    for p in si.timing_paths:
        if p.f_clk_hz is None:
            continue
        period = 1.0 / float(p.f_clk_hz.value)
        frac = float(p.capture_fraction.value) if p.capture_fraction is not None else None
        out += [f"- **타이밍 경로 {p.name}** (`{p.clock_net}` → {', '.join(f'`{n}`' for n in p.data_nets)}):", "",
                f"      T = 1/f_clk = 1/{quantity(float(p.f_clk_hz.value), 'Hz')} = {quantity(period, 's')},  t_lc = {frac if frac is not None else NO_RECORD} × T = "
                + (quantity(frac * period, 's') if frac is not None else NO_RECORD),
                "      setup = t_lc + t_flight(clk) − t_co,max − t_flight(data) − t_su,min ≥ 0",
                "      hold  = (T − t_lc) + t_co,min + t_flight(data) − t_flight(clk) − t_h,min ≥ 0",
                "      t_flight ∈ [0, 넷 전체의 배선 지연]  (발사 핀에서 캡처 핀까지의 경로를 특정하지 않으므로 구간; 최악의 경우 ≥ 0 이면 PASS, 최선의 경우도 < 0 이면 FAIL)", ""]
    return out


# --------------------------------------------------------------------------- promotions and the per-net table


def promotion_lines(ir: CircuitIR) -> list[str]:
    st = _st()
    si = ir.si
    assert si is not None
    out = ["### 임계 길이 규칙과 승격된 넷", "",
           "첫 배선 뒤 각 넷의 **선로** — 구리와 라이브러리의 패드 상자로 뽑은 가장 긴 패드-패드 경로(트랙 + 지나는 비아 배럴), 패드가 둘인 넷이면 그 구리 전체 — 의 지연을 재어, "
           "지연이 f × t_r 을 넘는 넷(= 선로 길이 > l_crit)을 '전기적으로 긴' 넷으로 봅니다. 가지를 모두 더한 넷 전체의 구리는 어떤 경로보다도 긴 상한일 뿐이라, "
           "경로를 뽑지 못한 넷(패드가 셋 이상)은 전체 구리로도 짧을 때만 '짧음' 이고, 그렇지 않으면 '길 수도 있음'(NOT_VERIFIED, 승격하지 않음)입니다. "
           "그 넷의 클래스가 `promote_to` 를 말하면 그 제어 임피던스 클래스로 **승격** 하고(기록: `ir.si` 의 `promoted`, provenance `derived` / `si.promote`), "
           "승격이 라우터 규칙을 바꾸면 보드를 **한 번** 다시 배선합니다. 전원·접지 넷은 구동 에지가 없어 규칙에서 빠지고, 승격도 재배선도 한 번뿐이라 끝없이 반복되지 않습니다.", ""]
    rows: list[list[object]] = []
    for c in si.net_classes:
        for p in c.promoted:
            widths = sorted({float(t.width_mm) for t in (ir.pcb.tracks if ir.pcb is not None else []) if t.net == p.net})
            rows.append([f"`{p.net}`", f"`{p.from_class}` → `{c.name}`", f"{p.length_mm:.3f}", f"{p.delay_s * 1e12:.2f}", f"{p.l_crit_mm:.3f}", quantity(p.t_rise_s, "s"),
                         ", ".join(f"{w:g}" for w in widths) or "-", p.provenance.note or NO_RECORD])
    if rows:
        out += [st._table(["넷", "승격", "첫 배선 선로 길이 (mm)", "지연 (ps)", "l_crit (mm)", "t_r", "지금 트랙 폭 (mm)", "근거 (provenance 메모 그대로)"], rows), ""]
        stack = ir.pcb.stackup if ir.pcb is not None else None
        planes = stack is not None and all(line_geometry(stack, layer)[0] is not None for layer in ROUTING_LAYERS)
        if not planes:
            out += ["이 보드에는 라우팅 층 옆에 기준 평면이 없어 승격된 넷도 제어 폭을 받지 못했습니다(임피던스가 정의되지 않음). 그래서 라우터 규칙이 바뀌지 않았고 다시 배선하지 않았습니다: "
                    "위 넷들은 '임피던스 제어가 필요한' 넷으로 기록만 되며, `si.impedance` / `spice.si` 는 NOT_VERIFIED "
                    "\"impedance is undefined without a reference plane - use pcb_layers=4 or add a plane\" 입니다.", ""]
        else:
            out += ["기준 평면이 있는 보드라 승격된 넷은 제어 폭으로 한 번 다시 배선되었습니다(위의 '지금 트랙 폭'). 그 폭의 임피던스는 `si.impedance`, "
                    "그 넷의 반사·링잉은 `spice.si` 가 판정합니다.", ""]
    else:
        out += ["승격된 넷 없음: 첫 배선에서 임계 길이를 넘은 넷이 없었거나(모두 전기적으로 짧음), 넘은 넷의 클래스가 승격 대상을 말하지 않았습니다.", ""]
    crit = ir.validation.latest(_CRITICAL)
    if crit is not None and isinstance(crit.details, dict):
        promoted = {p.net for c in si.net_classes for p in c.promoted}
        long_now = [str(n) for key in ("long_over_plane", "long_without_plane") for n in crit.details.get(key) or []]
        fresh = [n for n in long_now if n not in promoted and (si.class_of(n) is None or si.class_of(n).target_z0_ohm is None)]  # type: ignore[union-attr]
        if fresh:
            out += [f"재배선 뒤의 구리에서 새로 길어진 넷: {', '.join(f'`{n}`' for n in fresh)}. 승격과 재배선은 한 번뿐이므로 이 넷은 승격되지 않았고, "
                    f"`{_CRITICAL}` 이 목록에 올려 `spice.si` 가 (기준 평면이 있으면) 시뮬레이션으로 판정합니다.", ""]
    return out


def net_table_lines(ir: CircuitIR, figures: ReportFigures) -> list[str]:
    st = _st()
    crit = ir.validation.latest(_CRITICAL)
    out = ["### 넷별 길이·지연 (`si.critical_length` 기록 그대로)", ""]
    if crit is None:
        return out + [f"`{_CRITICAL}` {NO_RECORD}: IR_BUILD 단계의 검증이 아직 없거나 배치된 보드가 없습니다.", ""] + figures.lines(SLOT_SI_DELAY)
    out += [_result_line(crit), ""]
    rows = crit.details.get("nets") if isinstance(crit.details, dict) else None
    from ai_eda.validation.si import SHORT_TEXT

    # the rule's stock sentence for a short net, said once in the result line above; every other reason is printed as recorded
    short_words = {SHORT_TEXT: "짧음 (규칙에 의한 판단, 시뮬레이션 아님)", SHORT_TEXT + " (with the no-plane upper bound of t_pd)": "짧음 (t_pd 상한으로도; 규칙에 의한 판단)"}
    routed: list[list[object]] = []
    skipped: list[str] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        if _num(row.get("length_mm")) is None:
            skipped.append(f"`{row.get('net')}` ({st._cell(row.get('reason') or row.get('status'))})")
            continue
        ends = row.get("ends")
        line = "–".join(f"`{e}`" for e in ends) if isinstance(ends, list) and len(ends) == 2 else ("넷 전체" if row.get("measure") else "-")
        routed.append([f"`{row.get('net')}`", row.get("class") or "-", line, f"{float(row['length_mm']):.3f}",
                       f"{float(row['delay_ps']):.2f}" if _num(row.get("delay_ps")) is not None else "-",
                       f"{float(row['t_pd_ps_per_mm']):.4f}" + (" (상한)" if row.get("bound") else "") if _num(row.get("t_pd_ps_per_mm")) is not None else "-",
                       f"{float(row['l_crit_mm']):.3f}" if _num(row.get("l_crit_mm")) is not None else "-", row.get("status", "?"),
                       short_words.get(str(row.get("reason")), row.get("reason") or "-")])
    if routed:
        out += [st._table(["넷", "클래스", "선로 (패드–패드)", "선로 길이 (mm)", "지연 (ps)", "t_pd (ps/mm)", "l_crit (mm)", "행 판정", "이유 (기록)"], routed), "",
                "'선로' 가 패드 둘이면 그 두 패드 사이의 가장 긴 경로를, '넷 전체' 면 패드가 둘인 넷의 구리 전체(또는 경로를 뽑지 못한 넷의 전체 구리, 상한)를 잰 것입니다. "
                "'행 판정' 이 NOT_APPLICABLE 이고 이유가 'judged by spice.si' 인 넷은 규칙이 길다고 찾은 넷이며, 판정은 아래의 `spice.si.<넷>` 입니다.", ""]
    if skipped:
        out += ["규칙에서 빠진 넷: " + ", ".join(skipped) + ".", ""]
    out += figures.lines(SLOT_SI_DELAY)
    return out


def impedance_lines(ir: CircuitIR, figures: ReportFigures) -> list[str]:
    st = _st()
    si = ir.si
    assert si is not None
    out = ["### 임피던스 (`si.impedance.<클래스>` 기록)", ""]
    for c in si.net_classes:
        if c.target_z0_ohm is None:
            continue
        r = ir.validation.latest(f"si.impedance.{c.name}")
        if r is None:
            out += [f"- `si.impedance.{c.name}`: {NO_RECORD}", ""]
            continue
        out += [_result_line(r), ""]
        segs = r.details.get("segments") if isinstance(r.details, dict) else None
        rows = [[f"`{s.get('net')}`", s.get("layer", "-"), s.get("width_mm", "-"), f"{float(s['length_mm']):.3f}" if _num(s.get("length_mm")) is not None else "-",
                 f"{float(s['z0_ohm']):.2f}" if _num(s.get("z0_ohm")) is not None else "-", s.get("reference") or "-", s.get("status", "?"), s.get("reason") or "-"]
                for s in segs or [] if isinstance(s, dict)]
        if rows:
            out += [st._table(["넷", "층", "폭 (mm)", "길이 (mm)", "Z0 (Ω)", "기준 평면", "판정", "이유 (기록)"], rows), ""]
    out += figures.lines(SLOT_SI_Z0)
    return out


def budget_lines(ir: CircuitIR) -> list[str]:
    """``si.width`` / ``si.length`` / ``si.delay`` / ``si.skew`` / ``si.diff`` results that judged something (not NOT_APPLICABLE), with their rows, copied."""
    st = _st()
    out = ["### 폭·길이·지연 예산, 스큐, 차동 쌍", ""]
    shown = 0
    na: list[str] = []
    for r in _results(ir):
        if not r.check_id.startswith(("si.width.", "si.length.", "si.delay.", "si.skew.", "si.diff.")):
            continue
        if r.status is ValidationStatus.NOT_APPLICABLE:
            na.append(f"`{r.check_id}`")
            continue
        shown += 1
        out.append(_result_line(r))
        details = r.details if isinstance(r.details, dict) else {}
        rows = [x for x in (details.get("nets") or details.get("rows") or []) if isinstance(x, dict)] + [x for x in details.get("unrouted") or [] if isinstance(x, dict)]
        for row in rows[:40]:
            if isinstance(row, dict) and row.get("reason"):
                out.append(f"  - {row.get('net', '')}: {row.get('status', '?')}: {st._cell(row.get('reason'))}".replace(" : ", " "))
    if not shown:
        out.append("판정한 예산 검사가 없습니다.")
    if na:
        out += ["", f"해당 없음 (클래스가 그 예산을 말하지 않거나, 구리가 필요한 구성 넷이 없음): {', '.join(na)}."]
    out.append("")
    return out


def timing_lines(ir: CircuitIR) -> list[str]:
    st = _st()
    out = ["### 타이밍 (`si.timing.<경로>` 기록)", ""]
    rs = [r for r in _results(ir) if r.check_id.startswith("si.timing.")]
    if not rs:
        return out + ["타이밍 경로 검사 결과가 없습니다 (선언된 경로가 없거나 검증 전).", ""]
    for r in rs:
        out.append(_result_line(r))
        d = r.details if isinstance(r.details, dict) else {}
        for m in d.get("missing") or []:
            out.append(f"  - 빠진 항: {st._cell(m)}")
        for k, v in (d.get("flight_upper_ps") or {}).items():
            out.append(f"  - `{k}` 비행 시간 상한 (넷 전체의 배선 지연): {v} ps")
        for row in d.get("data") or []:
            if isinstance(row, dict):
                out.append(f"  - `{row.get('data_net')}`: setup 여유 {row.get('setup_margin_ns')} ns, hold 여유 {row.get('hold_margin_ns')} ns → {row.get('status')}")
    out += ["", "데이터시트의 타이밍 항(t_co, t_su, t_h)은 데이터시트 사실 grounding(`parts datasheet-facts` 흐름)으로만 IR 에 들어옵니다. 오프라인에서는 없으므로 "
            "이 경로는 빠진 항을 이름으로 적은 NOT_VERIFIED 이며, 추정값으로 채우지 않습니다. 커넥터에 붙은 항(예: `J2.t_co`)은 커넥터 뒤의 보드 밖 부품(ISP 프로그래머)의 값이므로, "
            "온라인이 된다고 채워지지 않습니다: 그 부품 자신의 문서를 `--datasheet-url J2=<문서>` 로 주고 데이터시트 사실 파일로 grounding 해야 합니다.", ""]
    return out


def spice_lines(ir: CircuitIR, figures: ReportFigures) -> list[str]:
    st = _st()
    out = ["### 반사·링잉 시뮬레이션 (`spice.si.<넷>`, ngspice)", "",
           "`si.critical_length` 가 전기적으로 긴 넷으로 올린 넷마다, 기준 평면 위라서 Z0 와 t_d 가 정의되면 작은 덱을 컴파일해 ngspice 로 과도 해석합니다:", "",
           "      소스 ─R_s─ A ═══ T(Z0, t_d) ═══ B ─C_L─ 0     (0 → 1 V 펄스, 상승·하강 t_r)", "",
           "Z0 는 선로에서 구리가 가장 많은 폭의 `calc.tline.microstrip.z0`, t_d 는 선로(패드가 둘인 넷의 구리)의 배선 지연(neck-down 은 합침, 문서화된 단순화)입니다. "
           "덱은 드라이버 하나·선로 하나·수신단 하나이므로, 패드가 셋 이상인 넷(가지와 다른 부하), 덱이 모델하지 않는 부품(저항·커패시터·인덕터·스위치·크리스털 등)이 달린 넷, "
           "클래스의 드라이버가 그 넷에서 입력 핀뿐인 넷은 시뮬레이션하지 않고 NOT_VERIFIED 로 그 이유를 적습니다(IR 에 없는 회로에 대한 판정을 만들지 않음). "
           "먼 끝 B 의 오버슈트·언더슈트가 스윙의 대역(클래스의 `ringing_tol_rel`) 안이고 각 레벨의 마지막 10 % 동안 대역 안에 머물면 PASS 입니다. "
           "덱과 rawfile 은 증거로 해시와 함께 기록됩니다(rawfile 에는 ngspice 의 Date 줄이 있어 결정론적 산출물이 아닌 증거).", ""]
    rs = [r for r in _results(ir) if r.check_id.startswith("spice.si.")]
    if not rs:
        out += ["`spice.si` 결과 없음: 전기적으로 긴 넷이 없었거나 SPICE 단계 전입니다.", ""]
    live = [r for r in rs if r.status is not ValidationStatus.NOT_APPLICABLE]
    old = [r for r in rs if r.status is ValidationStatus.NOT_APPLICABLE]
    # results that differ only in their net's name (every long net of a board without a plane) are one line
    grouped: dict[tuple[str, str], list[ValidationResult]] = {}
    for r in live:
        net = r.check_id.removeprefix("spice.si.")
        if r.status is ValidationStatus.NOT_VERIFIED:
            grouped.setdefault((str(r.status), r.message.replace(net, "<넷>")), []).append(r)
    for (status, text), group in grouped.items():
        if len(group) > 1:
            out.append(f"- {', '.join(f'`{r.check_id}`' for r in group)}: **{status}** {_tool(group[0])} — {_message(text)}")
    many = {r.check_id for group in grouped.values() if len(group) > 1 for r in group}
    for r in sorted(live, key=lambda r: (_STATUS_ORDER.get(r.status, 9), r.check_id)):
        if r.check_id in many:
            continue
        out.append(_result_line(r))
        d = r.details if isinstance(r.details, dict) else {}
        m = d.get("measured") if isinstance(d.get("measured"), dict) else None
        if m is not None:
            out.append(f"  - 측정 (기록): 최대 {float(m.get('max_v', float('nan'))):.4f} V, 최소 {float(m.get('min_v', float('nan'))):.4f} V, 오버슈트 {float(m.get('overshoot_rel', 0)):.2%}, "
                       f"언더슈트 {float(m.get('undershoot_rel', 0)):.2%}, 대역 ±{float(m.get('band_rel', 0)):.0%}")
        if d.get("deck"):
            out += ["", "  덱 (기록 그대로):", "", *("      " + line for line in str(d["deck"]).rstrip("\n").split("\n")), ""]
        ev = [e.path.replace("\\", "/").rsplit("/", 1)[-1] for e in r.evidence if e.path]
        if ev:
            out.append(f"  - 증거: {', '.join(f'`{n}`' for n in ev)}")
    if old:
        out.append(f"- 더 이상 길지 않아 대체된 결과 (NOT_APPLICABLE): {', '.join(f'`{r.check_id}`' for r in old)}")
    out.append("")
    out += figures.lines(SLOT_SI_STEP)
    return out


def limit_lines() -> list[str]:
    return [
        "### 정직한 한계", "",
        "- 모든 선로 식은 준정적(TEM)·무손실입니다: 분산, 도체·유전 손실, 표면 거칠기, 솔더 마스크(Z0 를 낮춤)는 모델에 없습니다. **실제 임피던스는 fab 의 측정(쿠폰)만이 말합니다** — "
        "이 보고서의 Z0 는 적층 수치를 식에 넣은 값입니다.",
        "- 템플릿의 일반 적층(위 표)은 fab 의 값이 아니라 선택값입니다. `--fab-capability` 의 `stackup` 블록은 페이지에 대해 grounding 되고 보고되지만, "
        "이 버전에서는 `ir.pcb.stackup` 에 기록되지 않습니다(`mfg.capability_source` 가 'not recorded in the IR' 이라고 말함): 모든 수치는 위 표의 적층으로 계산했습니다.",
        "- 컴파일된 `.kicad_pcb` 에는 `(stackup)` 절이 없습니다(층 목록과 평면 존만 있음). KiCad 는 파일을 열 때 자기 기본 적층을 만들므로, KiCad 의 보드 설정·임피던스 계산기·"
        "gerber job 파일의 재료 적층은 위 표(`ir.pcb.stackup`)가 아니라 KiCad 기본값을 보여 줍니다.",
        "- 드라이버 모델(t_r, R_s, C_L)은 보수적인 선택값이며 IBIS 모델이 아닙니다. 드라이버 부품의 데이터시트 t_rise / r_out / c_in 이 grounding 되면 그 값이 대신 쓰입니다.",
        "- `spice.si` 는 패드가 둘인 넷 하나를 무손실 선 하나로 봅니다(neck-down 을 합침). 가지가 있는 넷, 수동 부품이 달린 넷, 드라이버가 구동하지 않는 넷은 판정하지 않습니다(NOT_VERIFIED). "
        "종단 저항을 넣는 것 같은 설계 변경은 사람의 결정이며 파이프라인이 하지 않습니다.",
        "- 넷 클래스 폭은 IR 기하 검사(`si.width` / `si.impedance`)로만 확인합니다. 컴파일된 `.kicad_pro` 는 넷 클래스를 싣지 않으므로 **KiCad DRC 의 넷 클래스 폭 검사는 측정되지 않았습니다**.",
        "- 평면은 KiCad 가 채우는 존입니다. 채움 결과(다른 넷 둘레의 빈틈, 평면의 연속성)는 kicad-cli 가 있어야 확인되며, 이 보고서의 평면 그림은 존의 외곽선입니다.",
        "",
    ]


def si_circuit_section(ir: CircuitIR, figures: ReportFigures) -> list[str]:
    """The circuit report's "임피던스·타이밍" section (module docstring)."""
    out = ["## 임피던스·타이밍", ""]
    si = ir.si
    if si is None:
        return out + [NO_SI + ".", ""]
    out += [
        "이 절은 `ir.si`(넷 클래스·타이밍 경로), `ir.pcb.stackup`, 배선된 구리, 그리고 `si.*` / `spice.si.*` 검사 기록을 옮긴 것입니다. 신호 무결성 처리는 **필요한 곳에서만** 합니다: "
        "템플릿이 회로에 필요하다고 선언한 것(예: 크리스털 루프 길이, 전원 레일 최소 폭, ISP 타이밍 경로)과, 첫 배선 뒤 임계 길이 규칙이 '전기적으로 긴' 넷으로 찾은 넷만 제약을 받습니다. "
        + router_version_text(ir) + " "
        "`si.*` 는 IR 기하 + 등록된 계산기의 검사이며 DRC 가 아니고, `spice.si.*` 는 ngspice 의 도구 증거입니다.", "",
        "### 적층 (stackup)", "",
    ]
    out += stackup_lines(ir.pcb.stackup if ir.pcb is not None else None)
    out += ["### 넷 클래스와 타이밍 경로", ""]
    out += class_lines(ir)
    out += figures.lines(SLOT_SI_BOARD)
    out += formula_lines(ir)
    out += promotion_lines(ir)
    out += net_table_lines(ir, figures)
    out += impedance_lines(ir, figures)
    out += budget_lines(ir)
    out += timing_lines(ir)
    out += spice_lines(ir, figures)
    out += limit_lines()
    return out


# --------------------------------------------------------------------------- theory report


def si_theory_section(ir: CircuitIR) -> list[str]:
    """The theory report's transmission-line / timing section: the formulas in Korean with the IR's stack and class numbers (display values, no verdict)."""
    out = ["## 전송선로와 타이밍 (신호 무결성 이론)", ""]
    si = ir.si
    if si is None:
        return out + [NO_SI + ".", ""]
    stack = ir.pcb.stackup if ir.pcb is not None else None
    fraction = float(si.critical_fraction.value) if si.critical_fraction is not None else None
    ctrl = next((c for c in si.net_classes if c.target_z0_ohm is not None), None)
    dflt = si.default_class() or next((c for c in si.net_classes if c.t_rise_s is not None), None)
    t_r = float(dflt.t_rise_s.value) if dflt is not None and dflt.t_rise_s is not None else None
    r_s = float(dflt.r_drive_ohm.value) if dflt is not None and dflt.r_drive_ohm is not None else None
    c_l = float(dflt.c_load_f.value) if dflt is not None and dflt.c_load_f is not None else None
    band = float(dflt.ringing_tol_rel.value) if dflt is not None and dflt.ringing_tol_rel is not None else None
    z0 = float(ctrl.target_z0_ohm.value) if ctrl is not None and ctrl.target_z0_ohm is not None else None
    out += [
        "### 언제 배선이 '전송선' 이 되는가", "",
        "신호 에지가 배선을 한 번 왕복하는 시간보다 짧으면, 배선은 한 덩어리의 도선이 아니라 전송선로로 동작합니다. 드라이버가 보는 것은 배선의 특성 임피던스 Z0 이고, "
        "끝에서 반사된 파가 돌아와 오버슈트·언더슈트·링잉을 만듭니다. 이 시스템이 쓰는 판단 규칙은 다음과 같습니다(`calc.tline.critical_length`):", "",
        "      왕복 시간 2·l·t_pd < t_r   ⇔   l < l_crit = f × t_r / t_pd,   f = 1/2",
        "",
        f"f 는 확인된 선택값(`si.critical_fraction` = {fraction if fraction is not None else NO_RECORD})입니다. 더 엄격한 규칙은 f = 1/6(에지 길이의 1/6 보다 짧으면 집중 회로; "
        "Johnson & Graham, *High-Speed Digital Design*, 1993)입니다. 규칙은 시뮬레이션이 아니며, 규칙을 넘는 넷은 SPICE 로 넘깁니다.", "",
        "### 전파 지연", "",
        "      t_pd = √e_eff / c0,   c0 = 299 792 458 m/s,   t_d = l × t_pd",
        "",
        "e_eff 는 선로 주변 전기장이 보는 유효 유전율입니다. 기준 평면 위의 마이크로스트립에서는 e_eff 가 εr 보다 작고(전기장 일부가 공기 중), 기준 평면이 없으면 "
        "e_eff 를 정할 수 없어 상한 e_eff ≤ εr 만 씁니다(지연을 길게 보는 쪽).", "",
    ]
    if stack is not None:
        out += _design_lcrit_lines(ir, stack, fraction, t_r)
    out += [
        "### 특성 임피던스 (마이크로스트립, Hammerstad & Jensen 1980)", "",
        "      u = w/h,   Z01(u) = η0/(2π)·ln(f(u)/u + √(1 + (2/u)²)),   f(u) = 6 + (2π − 6)·exp(−(30.666/u)^0.7528)",
        "      e_eff(u) = (εr + 1)/2 + (εr − 1)/2·(1 + 10/u)^(−a(u)·b(εr))",
        "      a(u) = 1 + ln((u⁴ + (u/52)²)/(u⁴ + 0.432))/49 + ln(1 + (u/18.1)³)/18.7,   b(εr) = 0.564·((εr − 0.9)/(εr + 3))^0.053",
        "      두께 보정: Δu₁ = (t/h)/π·ln(1 + 4e/((t/h)·coth²√(6.517u))),  Δu_r = Δu₁·(1 + sech√(εr − 1))/2",
        "      Z0 = Z01(u + Δu_r)/√e_eff(u + Δu_r)",
        "",
        "η0 = μ0·c0 ≈ 376.73 Ω. 식이 밝힌 정확도(두께 0): Z01 은 u ≤ 1000 에서 0.03 % 이내, e_eff 는 0.01 ≤ u ≤ 100, εr < 128 에서 0.2 % 이내이며, 계산기는 그 범위 밖을 거부합니다. "
        "**임피던스는 기준 평면이 있을 때만 정의됩니다**: 2층 보드(F.Cu / 코어 / B.Cu)에는 트랙 옆에 연속된 평면이 없으므로 이 시스템은 숫자를 지어내지 않고 "
        "\"impedance is undefined without a reference plane - use pcb_layers=4 or add a plane\" 이라고 답합니다. 4층 보드(`pcb_layers=4`)는 In1.Cu 가 접지 평면, In2.Cu 가 전원 평면이고, "
        "F.Cu / B.Cu 의 트랙은 그 평면 위의 마이크로스트립입니다.", "",
    ]
    if stack is not None and z0 is not None:
        g, _why = line_geometry(stack, "F.Cu")
        if g is not None:
            h, t_um, er = float(g.h.value), float(g.t.value), float(g.er.value)
            from ai_eda.tools.calc.tline import solve_microstrip_width

            try:
                w = solve_microstrip_width(z0, h, t_um, er)
                r = microstrip(w, h, t_um, er)
                out += [f"이 설계의 F.Cu (h = {h:g} mm, t = {t_um:g} µm, εr = {er:g}, 기준 {g.reference_layer} {g.reference_net}): 목표 {z0:g} Ω 의 폭은 "
                        f"`calc.tline.width_for_z0.microstrip` 로 w = {w:.6f} mm (순방향 식의 이분법 해, 상대 오차 1e-6), 그때 e_eff = {r.e_eff:.4f}, "
                        f"t_pd = {propagation_delay(r.e_eff) * 1e9:.4f} ps/mm 입니다. 라우터는 이 폭을 0.01 mm 단위로 **올려서** 씁니다(좁으면 Z0 가 커짐).", ""]
            except (TLineRangeError, ValueError) as e:
                out += [f"목표 {z0:g} Ω 의 폭: 계산기가 거부 ({e}).", ""]
        else:
            out += ["이 설계의 적층에는 라우팅 층 옆 기준 평면이 없어, 목표 임피던스의 폭을 계산할 수 없습니다(제어 임피던스 클래스는 기록만 됨).", ""]
    out += [
        "### 반사와 링잉", "",
        "      Γ = (Z_L − Z0)/(Z_L + Z0),   Γ_s = (R_s − Z0)/(R_s + Z0)",
        "      가까운 끝의 첫 계단: V_A = V × Z0/(R_s + Z0)",
        "      먼 끝이 개방(C_L = 0, Γ_L = +1)이면 첫 도달: V_B = V_A × (1 + Γ_L) = 2·V_A",
        "      왕복할 때마다 링잉의 크기는 Γ_s × Γ_L 배",
        "",
    ]
    if r_s is not None and z0 is not None:
        gs = (r_s - z0) / (r_s + z0)
        va = z0 / (r_s + z0)
        out += [f"이 설계의 선택값 R_s = {r_s:g} Ω, Z0 = {z0:g} Ω: Γ_s = ({r_s:g} − {z0:g})/({r_s:g} + {z0:g}) = {gs:.4f}, V_A = {va:.4f} V (스윙 1 V 당), "
                f"개방 끝(C_L = 0)의 첫 도달 = {2 * va:.4f} V → 오버슈트 {(2 * va - 1) * 100:.1f} % 입니다. 이것은 **상한이 아닙니다**: 용량성 부하 "
                f"C_L = {quantity(c_l, 'F') if c_l is not None else NO_RECORD} 은 처음에 Γ ≈ −1 로 반사하다가 충전되며 +1 로 바뀌고, 그 반사가 소스에서 "
                "Γ_s < 0 으로 다시 반사되어 첫 도달 위에 겹치므로 봉우리를 이 값보다 높이거나 낮출 수 있습니다(유한한 t_r 은 낮춥니다). "
                "그래서 이 식은 크기를 가늠할 뿐이고, 판정은 `spice.si` 의 ngspice 과도 해석만 합니다. 판정 대역은 " + (f"±{band * 100:g} %" if band is not None else NO_RECORD) + " 입니다. "
                "직렬 종단(R_s + R_t = Z0)이면 첫 도달이 정확히 1 이 되어 오버슈트가 없어지지만, 부품을 더하는 것은 사람의 설계 결정입니다.", ""]
    out += [
        "### 넷 클래스와 '필요한 곳에서만'", "",
        "모든 넷을 제어 임피던스로 만들지 않습니다. 템플릿은 회로가 요구하는 것만 선언합니다(예: ATmega128 보드의 XTAL 루프 길이 예산, 전원 레일의 IPC-2221 최소 폭, ISP SPI 타이밍 경로). "
        "나머지 넷은 기본 클래스(`DEFAULT`: 임피던스 목표 없음, 보드 규칙으로 배선)이고, 첫 배선 뒤 임계 길이를 넘은 넷만 제어 임피던스 클래스로 승격되어 그 폭으로 한 번 다시 배선됩니다. "
        "승격은 `derived` provenance 에 규칙·측정 길이·지연·l_crit 을 적어 `ir.si` 에 기록됩니다.", "",
        "### 전류 용량 최소 폭 (IPC-2221, 외층)", "",
        "      I = k × ΔT^0.44 × A^0.725,   k = 0.048,   A [mil²] = w × t",
        "      → A = (I/(k × ΔT^0.44))^(1/0.725),   w_min = A / t   (`calc.ipc2221.width_for_current`)",
        "",
        "### 동기 인터페이스의 타이밍 여유", "",
        "      T = 1/f_clk,   t_lc = (발사→캡처 비율) × T",
        "      setup = t_lc + t_flight(clk) − t_co,max − t_flight(data) − t_su,min ≥ 0",
        "      hold  = (T − t_lc) + t_co,min + t_flight(data) − t_flight(clk) − t_h,min ≥ 0",
        "",
        "t_co, t_su, t_h 는 데이터시트 값이어야 합니다. grounding 되지 않은 항은 추정하지 않으며, 그 경로의 검사는 빠진 항을 이름으로 적은 NOT_VERIFIED 입니다. "
        "배선된 뒤의 비행 시간 t_flight 는 넷 전체의 배선 지연을 상한으로 하는 구간 [0, t_d] 로 넣습니다.", "",
        "### 시뮬레이션으로 확인하는 것", "",
        "임계 길이를 넘고 기준 평면 위에 있는 넷마다, ngspice 의 무손실 전송선 `T`(Z0, t_d)를 R_s 로 구동하고 먼 끝에 C_L 을 달아 계단 응답을 봅니다. "
        "이 덱은 드라이버 하나·선로 하나·수신단 하나이므로 그런 회로인 넷만 판정합니다: 패드가 셋 이상인 넷(가지), 덱이 모델하지 않는 부품(저항·커패시터·스위치 등)이 달린 넷, "
        "클래스의 드라이버가 그 넷을 구동하지 않는 넷(예: MCU 의 ~RESET 입력)은 NOT_VERIFIED 로 이유를 적습니다. "
        f"드라이버 에지 t_r = {quantity(t_r, 's') if t_r is not None else NO_RECORD}(보수적 선택값: 데이터시트 t_rise 가 grounding 되면 대체), R_s = {quantity(r_s, 'ohm') if r_s is not None else NO_RECORD}, "
        f"C_L = {quantity(c_l, 'F') if c_l is not None else NO_RECORD}. 결과와 그림은 회로 보고서의 '임피던스·타이밍' 절에 있습니다.", "",
    ]
    return out


def _design_lcrit_lines(ir: CircuitIR, stack: Stackup, fraction: float | None, t_r: float | None) -> list[str]:
    """The design's own t_pd and l_crit, per routing layer: the no-plane bound only where a layer has no reference plane (the value
    the rule then uses), the microstrip's per routed width where it has one (l_crit depends on the width through e_eff)."""
    out: list[str] = []
    widths = sorted(_widths(ir)) or []
    for layer in ROUTING_LAYERS:
        g, why = line_geometry(stack, layer)
        if g is None:
            d = stack.dielectrics[0 if layer == "F.Cu" else len(stack.dielectrics) - 1]
            er = float(d.er.value)
            tp = propagation_delay(er)
            text = f"이 설계의 {layer} 에는 기준 평면이 없습니다: 상한 t_pd = √{er:g} / c0 = {tp * 1e9:.4f} ps/mm"
            if fraction is not None and t_r is not None:
                text += (f", 그래서 l_crit ≥ {fraction:g} × {quantity(t_r, 's')} / {tp * 1e9:.4f} ps/mm = {critical_length_mm(t_r, tp, fraction):.2f} mm "
                         "(실제 l_crit 은 이보다 깁니다; 규칙은 이 하한 값을 그대로 써서 넷을 '길다' 고 보는 쪽으로 판단합니다)")
            out += [text + ".", ""]
            continue
        h, t_um, er = float(g.h.value), float(g.t.value), float(g.er.value)
        rows: list[str] = []
        for w in widths:
            try:
                r = microstrip(w, h, t_um, er)
            except (TLineRangeError, ValueError):
                continue
            tp = propagation_delay(r.e_eff)
            rows.append(f"w = {w:g} mm: e_eff = {r.e_eff:.4f}, t_pd = {tp * 1e9:.4f} ps/mm"
                        + (f", l_crit = {critical_length_mm(t_r, tp, fraction):.2f} mm" if fraction is not None and t_r is not None else ""))
        out += [f"이 설계의 {layer} 는 {g.reference_layer} ({g.reference_net} 평면) 위의 마이크로스트립입니다(h = {h:g} mm, εr = {er:g}): t_pd 와 l_crit 은 "
                "배선 폭에 따라 다릅니다(e_eff 가 폭에 따라 변함) — " + ("; ".join(rows) if rows else f"배선된 트랙 {NO_RECORD}") + ".", ""]
    return out


def _safe_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", text) or "net"


__all__ = [
    "NO_SI",
    "SI_PREFIXES",
    "SLOT_SI_BOARD",
    "SLOT_SI_DELAY",
    "SLOT_SI_STEP",
    "SLOT_SI_Z0",
    "si_circuit_section",
    "si_theory_section",
    "stackup_lines",
]
