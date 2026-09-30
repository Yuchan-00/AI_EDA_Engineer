"""The RF fixture figures of the circuit report: each network's S21 / S11 (dB) and S21 phase curves read from its ngspice rawfiles.

Invariant: these are *views* under the rule of :mod:`ai_eda.report.figures`
(no status computed, nothing registered, saved or mutated; the same IR and
the same files give byte-identical SVG). A curve is drawn only from
evidence about the current design:

* the network's recorded summary ``spice.rf.<network>`` must be tool-backed
  (a deck ran), not superseded (``NOT_APPLICABLE``) and stamped with the
  current design hash (:func:`fixture_freshness`);
* every rawfile a chart reads must be named in that summary's ``evidence``
  and be on disk with the recorded content hash
  (:func:`recorded_rawfile`) - a rawfile changed or removed since is no
  evidence, and the chart is left out with the reason.

What a chart shows: the complex node voltages ngspice wrote for one sweep
analysis of the network (the runner's own single-point analyses
``rf_at_<k>`` are not curves), turned into the quantities the fixture
runner reads with the runner's own formulas
(:func:`ai_eda.tools.spice.rf_fixture.s_parameter`: ``S21 = 2 V_to / V_s *
sqrt(R_drive / R_to)``, ``S11 = 2 V_drive / V_s - 1``; to a probe the voltage
ratio ``2 V_to / V_s``) from the port nodes, reference nodes and reference
impedances the runner recorded for each deck (``details["decks"]``) - a
picture of the rawfile, never a new reading: the judged numbers are the
recorded rows' own ``details["measured"]`` (``levels_db["at"]`` for a
relative row), drawn as markers coloured by the recorded status. A sample
whose |S| is exactly 0 (a level of -inf dB) cannot be drawn and is counted
in the caption. At most :data:`~ai_eda.report.figures.MAX_SERIES` curves
share a chart; more are split over several charts.
"""

from __future__ import annotations

import cmath
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_eda.ir import CircuitIR, ValidationResult, ValidationStatus
from ai_eda.ir.rf import POINT_ANALYSIS_PREFIX
from ai_eda.report.figures import COLUMN_PX, MAX_SERIES, STATUS_COLOURS, STATUS_OTHER, Figure, Marker, Series, svg_line_chart
from ai_eda.tools.spice.rawfile import parse as parse_rawfile
from ai_eda.tools.spice.rf_fixture import CHECK_PREFIX, PHASE21_DEG, REL_S21_DB, S11_DB, S21_DB, level_db, phase_deg, s_parameter

__all__ = [
    "FIG_PREFIX",
    "LEVEL",
    "PHASE",
    "Curve",
    "SweepCurves",
    "deck_curve",
    "fixture_freshness",
    "fixture_summary",
    "network_curves",
    "recorded_rawfile",
    "s_parameter_figures",
]

#: the two kinds of chart: a level in dB (S21, S11, the probe voltage ratio) and the S21 phase in degrees
LEVEL, PHASE = "level", "phase"
#: the figure-id prefix of every S-parameter chart (``rf_s21_<network>_<analysis>_<kind>[_<k>]``)
FIG_PREFIX = "rf_s21"
#: a sweep this wide (f_max / f_min) is drawn on a log frequency axis when its command does not say
LOG_SPAN = 10.0


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


# --------------------------------------------------------------------------- which evidence may be drawn


def fixture_summary(ir: CircuitIR, network_id: str) -> ValidationResult | None:
    """The latest recorded summary ``spice.rf.<network_id>``, else ``None``."""
    return ir.validation.latest(f"{CHECK_PREFIX}.{network_id}")


def fixture_freshness(ir: CircuitIR, result: ValidationResult | None) -> str | None:
    """``None`` when ``result`` (a ``spice.rf.<network>`` summary) is evidence about the current design, else why not (Korean)."""
    if result is None:
        return "spice.rf 기록 없음 (SPICE 단계 전이거나 이 회로망이 시뮬레이션되지 않음)"
    if result.status is ValidationStatus.NOT_APPLICABLE:
        return f"{result.check_id} 은 대체된 결과(NOT_APPLICABLE)입니다"
    if not result.is_tool_backed:
        return f"{result.check_id} 은 도구 기록이 없어(덱이 시뮬레이션되지 않음) 곡선이 없습니다"
    if result.ir_hash != ir.content_hash():
        return f"{result.check_id} 은 이전 IR 버전의 결과입니다"
    return None


def recorded_rawfile(result: ValidationResult, path: str | None) -> Path | str:
    """The rawfile ``path`` when ``result`` names it as evidence and it is on disk with the recorded hash, else why not (Korean)."""
    if not path:
        return "rawfile 이 기록되지 않았습니다"
    ev = next((e for e in result.evidence if e.path == path), None)
    name = _basename(path)
    if ev is None:
        return f"rawfile {name} 이 {result.check_id} 의 증거 목록에 없습니다"
    p = Path(path)
    if not p.is_file():
        return f"rawfile {name} 이 디스크에 없습니다"
    if ev.content_hash is None or _sha256(p) != ev.content_hash:
        return f"rawfile {name} 이 기록된 해시와 다릅니다"
    return p


# --------------------------------------------------------------------------- curves


@dataclass
class Curve:
    """One drawn quantity of one deck over one sweep: S21 ``drive -> to`` (level or phase), S11 at ``drive``, or the probe voltage ratio."""

    kind: str
    quantity: str
    state: str | None
    drive: str
    to: str | None
    probe: bool
    label: str
    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    #: samples with |S| exactly 0 (a level of -inf dB): not drawable
    zeros: int = 0
    #: samples with no S-parameter at all (a zero source voltage, a non-finite vector value): not drawable
    unreadable: int = 0


def _label(quantity: str, state: str | None, drive: str, to: str | None, probe: bool) -> str:
    where = f" [{state}]" if state else ""
    if quantity == S11_DB:
        return f"S11 {drive}{where}"
    if quantity == PHASE21_DEG:
        return f"∠S21 {drive}→{to}{where}" if not probe else f"∠(2V/V_s) {drive}→{to} (프로브){where}"
    return f"S21 {drive}→{to}{where}" if not probe else f"2V/V_s {drive}→{to} (프로브){where}"


def _wanted(network: Any) -> list[tuple[str, str, str | None, str, str | None, bool]]:
    """``(kind, quantity, state, drive, to, probe)`` per distinct curve the network's rows read, in row order (expectations, then probes)."""
    ports = {p.name: p for p in network.ports}
    out: list[tuple[str, str, str | None, str, str | None, bool]] = []
    for row in [*network.expectations, *network.probes]:
        if row.quantity == S11_DB:
            key = (LEVEL, S11_DB, row.state, row.drive, None, False)
        else:
            to_port = ports.get(row.to)
            probe = to_port is not None and to_port.kind == "probe"
            kind = PHASE if row.quantity == PHASE21_DEG else LEVEL
            key = (kind, PHASE21_DEG if kind == PHASE else S21_DB, row.state, row.drive, row.to, probe)
        if key not in out:
            out.append(key)
    return out


def _vectors(path: Path) -> tuple[list[float], dict[str, list[float]]]:
    plot = parse_rawfile(path).as_plot_vectors()
    lower = {k.lower(): v for k, v in plot.items()}
    freq = lower.get("frequency")
    if not freq:
        raise ValueError(f"rawfile {path.name} has no frequency scale (not an ac analysis)")
    return list(freq), lower


def _voltage(vectors: dict[str, list[float]], node: str | None, k: int) -> complex:
    if node is None:
        return 0j
    re_, im_ = vectors.get(f"{node.lower()}.real"), vectors.get(f"{node.lower()}.imag")
    if re_ is None or im_ is None:
        raise ValueError(f"no complex vector {node!r} in the rawfile")
    return complex(re_[k], im_[k])


def _pair(value: Any) -> tuple[str | None, str | None]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (None if value[0] is None else str(value[0]), None if value[1] is None else str(value[1]))
    raise ValueError(f"a recorded node pair must be [node, reference], got {value!r}")


def deck_curve(deck: dict, freq: list[float], vectors: dict[str, list[float]], want: tuple[str, str, str | None, str, str | None, bool]) -> Curve:
    """One curve of a deck from its rawfile vectors, with the node map and reference impedances the runner recorded (``details["decks"][*]``)."""
    kind, quantity, state, drive, to, probe = want
    nodes = deck.get("nodes") or {}
    r_port = deck.get("r_port") or {}
    src_n, src_r = _pair(deck.get("source"))
    drv_n, drv_r = _pair(nodes.get(drive))
    to_n, to_r = _pair(nodes.get(to)) if to is not None else (None, None)
    r_drive = _num(r_port.get(drive))
    r_to = None if to is None or probe else _num(r_port.get(to))
    if r_drive is None or (to is not None and not probe and r_to is None):
        raise ValueError(f"no reference impedance recorded for port {drive if r_drive is None else to!r}")
    curve = Curve(kind, quantity, state, drive, to, probe, _label(quantity, state, drive, to, probe))
    for k, f in enumerate(freq):
        v_s = _voltage(vectors, src_n, k) - _voltage(vectors, src_r, k)
        v_drive = _voltage(vectors, drv_n, k) - _voltage(vectors, drv_r, k)
        v_to = (_voltage(vectors, to_n, k) - _voltage(vectors, to_r, k)) if to is not None else 0j
        try:
            s = s_parameter(S11_DB if quantity == S11_DB else S21_DB, v_s=v_s, v_drive=v_drive, v_to=v_to, r_drive=r_drive, r_to=r_to)
        except ValueError:  # a zero source voltage: no S-parameter
            curve.unreadable += 1
            continue
        if s == 0:  # an exact zero |S|: a level of -inf dB and no phase, not drawable
            curve.zeros += 1
            continue
        try:
            y = phase_deg(s) if kind == PHASE else level_db(s)
        except ValueError:  # a non-finite magnitude
            curve.unreadable += 1
            continue
        if not math.isfinite(y) or not math.isfinite(f):
            curve.unreadable += 1
            continue
        curve.xs.append(float(f))
        curve.ys.append(y)
    return curve


@dataclass
class SweepCurves:
    """Every curve of one network over one sweep analysis, with the rawfiles read and why a deck has none."""

    analysis_id: str
    command: str
    curves: list[Curve] = field(default_factory=list)
    rawfiles: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def network_curves(ir: CircuitIR, network: Any, summary: ValidationResult) -> list[SweepCurves]:
    """Per sweep analysis of ``network`` (IR order), the curves its rows read, from the rawfiles ``summary`` records (hash-checked)."""
    decks = summary.details.get("decks") if isinstance(summary.details, dict) else None
    wanted = _wanted(network)
    out: list[SweepCurves] = []
    for sweep in network.sweep:
        aid = sweep.id
        if aid.lower().startswith(POINT_ANALYSIS_PREFIX):
            continue
        sc = SweepCurves(aid, "")
        for deck in decks if isinstance(decks, list) else []:
            if not isinstance(deck, dict):
                continue
            mine = [w for w in wanted if w[2] == deck.get("state") and w[3] == deck.get("drive")]
            if not mine:
                continue
            key = str(deck.get("key") or deck.get("stem") or "?")
            info = (deck.get("analyses") or {}).get(aid)
            if not isinstance(info, dict) or not info.get("succeeded"):
                sc.missing.append(f"덱 {key}: 해석 {aid} 의 결과 없음")
                continue
            sc.command = sc.command or str(info.get("command") or "")
            raw = recorded_rawfile(summary, info.get("raw_output_path"))
            # every analysis writes a rawfile of the same name into its own folder: a reason names the analysis, a caption the folder too
            if isinstance(raw, str):
                sc.missing.append(f"덱 {key}, 해석 {aid}: {raw}")
                continue
            try:
                freq, vectors = _vectors(raw)
                curves = [deck_curve(deck, freq, vectors, w) for w in mine]
            except (ValueError, KeyError, IndexError, OSError) as e:
                sc.missing.append(f"덱 {key}, 해석 {aid}: rawfile {raw.name} 을 읽을 수 없습니다 ({e})")
                continue
            sc.rawfiles.append(f"{raw.parent.name}/{raw.name}")
            sc.curves += [c for c in curves if len(c.xs) >= 2]
            sc.missing += [f"{c.label}: 그릴 표본이 두 개 미만" for c in curves if len(c.xs) < 2]
        out.append(sc)
    return out


# --------------------------------------------------------------------------- markers (the recorded rows)


def _row_result(ir: CircuitIR, network: Any, row: Any) -> ValidationResult | None:
    sid = f".{row.state}" if row.state else ""
    return ir.validation.latest(f"{CHECK_PREFIX}.{network.id}{sid}.{row.id}")


def _marker_y(row: Any, result: ValidationResult) -> float | None:
    """The recorded number a row's marker is drawn at: ``measured`` (a phase moved to (-180, 180]), a relative row's recorded absolute level at ``at``."""
    d = result.details if isinstance(result.details, dict) else {}
    if row.quantity == REL_S21_DB:
        levels = d.get("levels_db") if isinstance(d.get("levels_db"), dict) else {}
        return _num(levels.get("at"))
    m = _num(d.get("measured"))
    if m is None:
        return None
    if row.quantity == PHASE21_DEG:
        return phase_deg(cmath.rect(1.0, math.radians(m)))
    return m


def _markers(ir: CircuitIR, network: Any, curves: list[Curve]) -> tuple[list[Marker], list[str]]:
    """Markers for the recorded expectation rows of the drawn curves (inside their sweep), and the rows left out with why."""
    keyed = {(c.kind, c.quantity, c.state, c.drive, c.to): c for c in curves}
    out: list[Marker] = []
    left: list[str] = []
    for row in network.expectations:
        kind = PHASE if row.quantity == PHASE21_DEG else LEVEL
        quantity = S11_DB if row.quantity == S11_DB else (PHASE21_DEG if kind == PHASE else S21_DB)
        curve = keyed.get((kind, quantity, row.state, row.drive, None if row.quantity == S11_DB else row.to))
        if curve is None:
            continue
        r = _row_result(ir, network, row)
        if r is None:
            left.append(f"{row.id}: 기록 없음")
            continue
        at = float(row.at.value)
        if not (min(curve.xs) <= at <= max(curve.xs)):
            left.append(f"{row.id} (f = {at:.10g} Hz, 이 스위프 밖)")
            continue
        y = _marker_y(row, r)
        if y is None:
            left.append(f"{row.id}: 기록된 수치 없음 ({r.status})")
            continue
        state = f"[{row.state}] " if row.state else ""
        out.append(Marker(at, y, f"{state}{row.id} {r.status}", STATUS_COLOURS.get(r.status.value, STATUS_OTHER)))
    return out, left


# --------------------------------------------------------------------------- the charts


def _log_axis(command: str, curves: list[Curve]) -> bool:
    words = command.split()
    if len(words) >= 2 and words[0].lower() == "ac":
        return words[1].lower() in ("dec", "oct")
    xs = [x for c in curves for x in c.xs if x > 0]
    return bool(xs) and max(xs) / min(xs) >= LOG_SPAN


def _ports_text(network: Any) -> str:
    parts = []
    for p in network.ports:
        if p.kind == "port" and p.z0_ohm is not None:
            parts.append(f"{p.name} {float(p.z0_ohm.value):g} Ω")
        elif p.kind == "probe":
            parts.append(f"{p.name} 프로브(부하 없음)")
    return ", ".join(parts) or "-"


def s_parameter_figures(ir: CircuitIR, network: Any, *, taken: set[str] | None = None, width: int = COLUMN_PX) -> tuple[list[Figure], list[str]]:
    """The S-parameter charts of one fixture network (module docstring) and the reasons for what is not drawn (Korean sentences).

    Nothing is drawn when the recorded summary is not fresh evidence
    (:func:`fixture_freshness`): the reason is returned instead. ``taken``
    holds the figure ids already used on the page (the new ids are added to
    it): an id that is taken gets a ``_`` appended, so two networks whose ids
    join to the same text never share a figure id or a clip path.
    """
    summary = fixture_summary(ir, network.id)
    why = fixture_freshness(ir, summary)
    if why is not None:
        return [], [why]
    assert summary is not None
    used = taken if taken is not None else set()
    figures: list[Figure] = []
    reasons: list[str] = []
    for sc in network_curves(ir, network, summary):
        reasons += sc.missing
        for kind in (LEVEL, PHASE):
            curves = [c for c in sc.curves if c.kind == kind]
            chunks = [curves[i:i + MAX_SERIES] for i in range(0, len(curves), MAX_SERIES)]
            for k, chunk in enumerate(chunks):
                fig_id = f"{FIG_PREFIX}_{network.id}_{sc.analysis_id}_{kind}" + (f"_{k + 1}" if len(chunks) > 1 else "")
                while fig_id in used:
                    fig_id += "_"
                used.add(fig_id)
                markers, left = _markers(ir, network, chunk)
                # a single curve has no legend: the title names it
                if len(chunk) == 1:
                    what = f"{chunk[0].label} " + ("크기 (dB)" if kind == LEVEL else "(°)")
                else:
                    what = "S-파라미터 크기 (dB)" if kind == LEVEL else "S21 위상 (°)"
                part = f" ({k + 1}/{len(chunks)})" if len(chunks) > 1 else ""
                title = f"{network.id}: {what} — 해석 {sc.analysis_id}{part} ({summary.status})"
                series = [Series(c.label, c.xs, c.ys) for c in chunk]
                svg = svg_line_chart(series, title=title, x_label="주파수 (Hz)", y_label="레벨 (dB)" if kind == LEVEL else "위상 (°)",
                                     log_x=_log_axis(sc.command, chunk), markers=markers, width=width, height=420, clip_id=fig_id)
                zeros = sum(c.zeros for c in chunk)
                unreadable = sum(c.unreadable for c in chunk)
                caption = (
                    f"{network.id} 의 고정구 덱이 해석 `{sc.analysis_id}`({sc.command or '명령 기록 없음'})에서 쓴 rawfile({', '.join(sc.rawfiles)})의 복소 노드 전압으로 "
                    "고정구 실행기와 같은 식을 계산해 그린 곡선입니다: S21 = 2·V_읽기/V_s·√(R_구동/R_읽기), S11 = 2·V_구동/V_s − 1, 프로브(부하 없음)는 전압비 2·V/V_s 로 "
                    f"그 위상·상대 레벨만 의미가 있습니다. 포트: {_ports_text(network)}. "
                    "점은 기록된 spice.rf 행의 측정값입니다(색 = 기록된 판정: 초록 PASS, 빨강 FAIL, 회색 그 외; 상대 레벨 행은 그 행이 기록한 at 의 절대 레벨). "
                    "판정은 기록된 결과의 것이며 그림은 새로 판정하지 않습니다. 확인된 모델값 아래의 회로도 수준 회로망(트랙·비아·접지 귀로 인덕턴스 없음)이며, "
                    "부품·보드·무전기의 측정이 아닙니다."
                    + (f" |S| = 0 인 표본 {zeros}개(−∞ dB)는 그릴 수 없어 뺐습니다." if zeros else "")
                    + (f" S-파라미터가 없는 표본 {unreadable}개(원천 전압 0 또는 유한하지 않은 값)는 뺐습니다." if unreadable else "")
                    + (f" 점을 찍지 않은 행: {', '.join(left)}." if left else "")
                )
                figures.append(Figure(fig_id, title, caption, svg))
    return figures, reasons
