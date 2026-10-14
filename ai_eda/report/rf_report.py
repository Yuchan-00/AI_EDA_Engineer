"""The RF text of the Korean stage reports: the UNVERIFIED-profile banner, the theory report's RF section, the circuit report's fixture section, the final report's RF summary.

Invariant: a *view* under the rule of :mod:`ai_eda.report.stages` - every
status printed is a stored ``ValidationResult.status`` copied verbatim (the
row statuses inside a result's ``details`` too), no verdict is computed,
nothing is registered, hashed or saved, no wall-clock and no absolute path
(evidence files by their path under the fixture folder - every analysis
writes a rawfile of one name into its own ``<stem>/<analysis>/`` folder, so
a basename would fold them together -, stored messages through
:func:`~ai_eda.report.stages.strip_paths`). The numbers are the IR's own
(``ir.rf``: the frequency plan, the fixture networks with their nominals,
bounds and tolerances, the lab items, the rail budgets, the blocks; the
``kr447.*`` profile and ``model.*`` values of ``ir.parameters`` with their
notes) and the recorded ``spice.rf.*`` / ``rf.*`` / ``block.interface.*`` /
``power.*`` / ``pcb.keepout`` results' details; the only arithmetic is a
display difference (a plan row's ``f - ref``), named as such. A design
without ``ir.rf`` gets none of it.

What the sections say, and where:

* :func:`profile_banner` - at the top of every report of a design that
  carries a regulatory profile (``ir.rf.profile_keys``, or any ``kr447.*``
  parameter): the profile is an UNVERIFIED user-confirmed placeholder, never
  a grounded fact, and ``rf.regulatory_profile`` never PASSes on it; a
  design-deck or fixture row whose target or frequency was derived from it
  may PASS, which says only that the circuit realises the design derived
  from the placeholder (never compliance) - the banner lists those rows from
  provenance (:func:`profile_dependents`); legality is not decided here; KC
  conformity assessment is needed before any transmission; RF performance is
  lab-only.
* :func:`rf_theory_section` (theory report) - the frequency plan (with the
  recorded ``rf.freq_plan`` row statuses), the profile table with the
  document each value would be grounded by, the model-value list (the
  ``rf.model_grounding`` rows copied), and every fixture network with what
  it judges (nominal and its origin, the tolerance or the one-sided bound)
  and the S-parameter formulas.
* :func:`rf_circuit_section` (circuit report) - the RF blocks (regions,
  chain, shield cans, the ``block.interface.*`` results) and keep-outs
  (``pcb.keepout``), then per fixture network its recorded summary, every
  row's recorded measurement and status, the probes, the decks and evidence
  files, and the S-parameter charts (:mod:`ai_eda.report.rf_figures`, read
  from the rawfiles only while the summary is about the current design and
  each rawfile has its recorded hash), then the ``rf.deviation`` chain.
* :func:`rf_final_section` (final report) - the latest RF results in one
  table, the lab list (``rf.lab.<id>``), the rail budgets (``power.*``) and
  the honest limits.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

from ai_eda.design.base import CHOICE_NOTE_PREFIX, NO_RECORD, UNIT_DISPLAY, quantity
from ai_eda.design.rf.models import MODEL_PREFIX
from ai_eda.design.rf.profile import CONFORMITY, PROFILE_PREFIX
from ai_eda.ir import CircuitIR, ProvenanceKind, Traced, ValidationResult, ValidationStatus
from ai_eda.ir.provenance import design_data
from ai_eda.tools.keepout import allowed_nets, forbids
from ai_eda.report.figures import bound_text
from ai_eda.report.rf_figures import fixture_summary, s_parameter_figures

if TYPE_CHECKING:
    from ai_eda.ir.rf import RFDesign, RFNetwork
    from ai_eda.report.stages import ReportFigures

#: the per-network figure slot of the circuit report (``rf_s21:<network id>``)
SLOT_RF_S21 = "rf_s21:"
#: the check ids the RF sections copy (the fixture runner's and the RF checks'), plus the keep-out check
FIXTURE_PREFIX = "spice.rf."
RF_CHECK_PREFIXES: tuple[str, ...] = ("rf.", "block.interface.", "power.rail_budget.", "power.headroom.")
FREQ_PLAN_CHECK = "rf.freq_plan"
PROFILE_CHECK = "rf.regulatory_profile"
MODEL_CHECK = "rf.model_grounding"
DEVIATION_CHECK = "rf.deviation"
LAB_PREFIX = "rf.lab."
KEEPOUT_CHECK = "pcb.keepout"
#: the heading of the banner (every report of a design with a regulatory profile starts with it)
BANNER_TITLE = "**[UNVERIFIED] 규제 프로파일: 검증되지 않은 자리표시값**"
#: a stored message longer than this is cut in the RF sections (the result itself carries all of it)
MESSAGE_CHARS = 600
#: rows of a result's ``details`` printed at most (the rest are counted)
MAX_ROWS = 60

_PLAN_KINDS = {"margin": "여유 (margin)", "response": "응답 (response)", "gated": "게이트 (gated)", "coincidence": "일치 금지 (coincidence)"}
_QUANTITIES = {"s21_db": "S21 (dB)", "s11_db": "S11 (dB)", "rel_s21_db": "상대 S21 (dB)", "phase21_deg": "S21 위상 (°)"}
_UNITS = {"s21_db": "dB", "s11_db": "dB", "rel_s21_db": "dB", "phase21_deg": "°"}
_UNVERIFIED_RE = re.compile(r"\[UNVERIFIED: ([^\]]*)\]")
#: the head of a template choice's note: ``design choice confirmed by user; template <id> v<version>: `` or its unconfirmed form
_CHOICE_HEAD_RE = re.compile(r"^(?:" + re.escape(CHOICE_NOTE_PREFIX) + r"; template \S+ v\S+|template \S+ v\S+ choice, not yet confirmed): ")
#: profile key suffix -> the requirement ``rf.regulatory_profile`` compares it with (``ai_eda.validation.rf.PROFILE_LIMITS`` and the band rows)
_PROFILE_COMPARES = {
    ".max_power": "`tx_power` 이하", ".max_deviation": "`frequency_deviation` 이하", ".max_obw": "`occupied_bandwidth` 이하",
    ".freq_tolerance": "`frequency_tolerance` 이하", ".band_low": "`carrier_frequency` 가 대역 안", ".band_high": "`carrier_frequency` 가 대역 안",
    ".channel_raster": "`carrier_frequency` 가 채널 간격 위",
}


def _st():
    """The stage-report helpers (imported lazily: :mod:`ai_eda.report.stages` imports this module)."""
    from ai_eda.report import stages

    return stages


def rf_of(ir: CircuitIR) -> RFDesign | None:
    """``ir.rf`` (``None`` for a design that states no RF content)."""
    return getattr(ir, "rf", None)


# --------------------------------------------------------------------------- small helpers


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _hz(f: float | None) -> str:
    """``447.5625 MHz``, ``21.4 MHz``, ``450 kHz`` - ten significant digits, enough to tell channels apart."""
    if f is None:
        return NO_RECORD
    for scale, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if abs(f) >= scale:
            return f"{f / scale:.10g} {unit}"
    return f"{f:.10g} Hz"


def _value(t: Traced | None) -> str:
    """A traced value for a table cell: a frequency to ten digits, a text verbatim (cut), a number with its unit."""
    if t is None:
        return NO_RECORD
    v = t.value
    if isinstance(v, str):
        first = v.strip().split("\n", 1)[0]
        return (first[:80] + " …") if len(first) > 80 or "\n" in v.strip() else first
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return str(v)
    if t.unit == "Hz":
        return _hz(float(v))
    unit = UNIT_DISPLAY.get(t.unit or "", t.unit or "")
    if t.unit in ("H", "F", "s", "A", "V", "ohm", "W"):
        return quantity(float(v), t.unit, 6)
    return f"{float(v):.10g}" + (f" {unit}" if unit else "")


def _level(v: Any, unit: str) -> str:
    """A recorded level / phase (a number, ``"-inf"`` / ``None`` for an exact zero)."""
    if v == "-inf":
        return f"−∞ {unit}"
    if v == "+inf":
        return f"+∞ {unit}"
    n = _num(v)
    return NO_RECORD if n is None else f"{n:.4f} {unit}"


def _origin(t: Traced | None) -> str:
    """How a reader should weigh a traced number: a confirmed template choice as such, else the provenance words (calculator id and version)."""
    if t is None:
        return NO_RECORD
    st = _st()
    p = t.provenance
    if p.kind is ProvenanceKind.USER_REQUIREMENT and (p.note or "").startswith(CHOICE_NOTE_PREFIX):
        return st.CHOICE_LABEL
    return st._provenance_text(p)


def _choice_body(note: str | None) -> str:
    """A template choice's note without its ``design choice confirmed by user; template <id> v<version>: `` head."""
    return _CHOICE_HEAD_RE.sub("", note or "", count=1)


def _grounding_source(note: str | None) -> str:
    """The ``[UNVERIFIED: ...]`` text of a choice's note (the document or measurement that would ground it), else :data:`NO_RECORD`."""
    found = _UNVERIFIED_RE.findall(note or "")
    return found[-1] if found else NO_RECORD


def _message(text: str) -> str:
    cell = _st()._cell(text)
    return cell if len(cell) <= MESSAGE_CHARS else cell[:MESSAGE_CHARS].rstrip() + " … (이하 생략: 검사 결과의 기록에 전부 있음)"


def _tool(r: ValidationResult) -> str:
    return f"({_st()._tool_text(r.tool, r.tool_version)})" if r.tool else "(도구 없음)"


def _result_line(ir: CircuitIR, r: ValidationResult | None, check_id: str) -> str:
    """``- `check`: **STATUS** (tool) [대상 IR] — message`` of a recorded result, copied; :data:`NO_RECORD` without one."""
    if r is None:
        return f"- `{check_id}`: {NO_RECORD}"
    fresh = _st()._freshness(r, ir.content_hash())
    return f"- `{r.check_id}`: **{r.status}** {_tool(r)} [{fresh}] — {_message(r.message)}"


def _rows(r: ValidationResult | None) -> list[dict]:
    rows = r.details.get("rows") if r is not None and isinstance(r.details, dict) else None
    return [x for x in rows if isinstance(x, dict)] if isinstance(rows, list) else []


def _more(n: int) -> list[str]:
    return [f"(나머지 {n - MAX_ROWS}개 행은 검사 결과의 기록에 있음)", ""] if n > MAX_ROWS else []


def _latest(ir: CircuitIR, prefixes: tuple[str, ...]) -> list[ValidationResult]:
    return [r for k, r in sorted(ir.validation.latest_by_check().items()) if k.startswith(prefixes)]


def profile_keys_of(ir: CircuitIR) -> list[str]:
    """The profile keys ``ir.rf.profile_keys`` lists, then any ``kr447.*`` parameter it does not (a report never hides one)."""
    rf = rf_of(ir)
    listed = list(rf.profile_keys) if rf is not None else []
    return listed + sorted(k for k in ir.parameters if k.startswith(PROFILE_PREFIX) and k not in listed)


def model_keys_of(ir: CircuitIR) -> list[str]:
    """The model values ``ir.rf.model_values`` lists, then any ``model.*`` parameter it does not (``rf.model_grounding`` names those too)."""
    rf = rf_of(ir)
    listed = list(rf.model_values) if rf is not None else []
    return listed + sorted(k for k in ir.parameters if k.startswith(MODEL_PREFIX) and k not in listed)


# --------------------------------------------------------------------------- the banner


def profile_dependents(ir: CircuitIR) -> list[str]:
    """The expectation checks whose nominal, frequency or analysis rests on a ``kr447.*`` placeholder, read from provenance alone.

    A value rests on the profile when it *is* a profile parameter's traced
    value (the same record copied, e.g. the raster an analysis runs at), or
    its provenance names - through ``derived_from`` / ``inputs``, followed
    through ``ir.parameters`` - a ``kr447.*`` key or such a copy. Nothing is
    computed about the circuit: this lists which recorded PASS / FAIL would
    move once the profile is grounded. Design-deck rows are ``spice.<id>``,
    fixture rows ``spice.rf.<network>[.<state>].<id>``.
    """
    params = ir.parameters
    profile = [design_data(t) for k, t in params.items() if k.startswith(PROFILE_PREFIX)]
    memo: dict[str, bool] = {}

    def rests(t: Traced | None) -> bool:
        if t is None:
            return False
        if design_data(t) in profile:  # the same record (its design view: a copy made at another instant is the same value)
            return True
        prov = t.provenance
        for key in [*prov.derived_from, *prov.inputs.values()]:
            if key.startswith(PROFILE_PREFIX):
                return True
            if key not in memo:
                memo[key] = False  # a cycle guard; overwritten below
                memo[key] = rests(params.get(key))
            if memo[key]:
                return True
        return False

    out: list[str] = []
    sim = ir.simulation
    if sim is not None:
        analyses = {a.id: a for a in sim.analyses}
        for e in sim.expectations:
            a = analyses.get(e.analysis_id)
            vals = [e.nominal, e.at, *((a.params or {}).values() if a is not None else [])]
            if any(rests(v) for v in vals if isinstance(v, Traced)):
                out.append(f"spice.{e.id}")
    rf = rf_of(ir)
    for n in rf.networks if rf is not None else []:
        for e in n.expectations:
            if any(rests(v) for v in (e.nominal, e.at, e.ref_at) if isinstance(v, Traced)):
                out.append(f"spice.rf.{n.id}" + (f".{e.state}" if e.state else "") + f".{e.id}")
    return out


def profile_banner(ir: CircuitIR) -> list[str]:
    """The UNVERIFIED-profile banner (module docstring); ``[]`` for a design without a regulatory profile."""
    keys = profile_keys_of(ir)
    if not keys:
        return []
    docs: list[str] = []
    for k in keys:
        t = ir.parameters.get(k)
        src = _grounding_source(t.provenance.note if t is not None else None)
        for doc in (d.strip() for d in src.split(";", 1)[0].split(" / ")):
            if src != NO_RECORD and doc and doc not in docs:
                docs.append(doc)
    unconfirmed = [k for k in keys if (t := ir.parameters.get(k)) is not None and t.provenance.kind is ProvenanceKind.ASSUMPTION]
    absent = [k for k in keys if k not in ir.parameters]
    conformity = ir.parameters.get(f"{PROFILE_PREFIX}conformity")
    dependents = profile_dependents(ir)
    r = ir.validation.latest(PROFILE_CHECK)
    out = [
        BANNER_TITLE, "",
        f"- 이 설계의 KR 447 MHz 규제 수치 {len(keys)}개({', '.join(f'`{k}`' for k in keys)})는 공식 본문을 보관·대조하지 못한 채 일반 지식으로 적은 값을 "
        "사용자가 확인한 **선택값(자리표시값)** 이며, grounding 된 사실이 아닙니다"
        + (f" (근거가 될 문서: {', '.join(docs)})" if docs else "") + ". `rf.regulatory_profile` 은 이 값으로 PASS 가 되지 않습니다: 최선이 NOT_VERIFIED 이고, "
        "확인된 두 값이 서로 어긋날 때만 FAIL 입니다(적법성 판정이 아님).",
        "- 이 값에서 목표·주파수를 얻은 설계 행(예: 채널 간격에서의 스플래터 감쇠, 편이 한계에서 정한 구동 한계)은 PASS 일 수 있습니다. 그 PASS 는 "
        "'회로가 자리표시값에서 유도한 설계를 구현한다' 는 뜻일 뿐 적합성 판정이 아니며, 프로파일이 grounding 되면 달라질 수 있습니다. "
        + (f"이 설계에서 출처(provenance)가 자리표시값에 닿는 행: {', '.join(f'`{c}`' for c in dependents)}." if dependents else
           "이 설계에는 출처가 자리표시값에 닿는 시뮬레이션 행이 없습니다."),
        "- 기록된 판정: " + (f"`{PROFILE_CHECK}` **{r.status}** {_tool(r)}" if r is not None else f"`{PROFILE_CHECK}` {NO_RECORD}") + ".",
    ]
    if unconfirmed:
        out.append(f"- 아직 사용자가 확인하지 않은(가정) 값: {', '.join(f'`{k}`' for k in unconfirmed)}.")
    if absent:
        out.append(f"- `ir.rf.profile_keys` 에 있으나 값이 없는 키: {', '.join(f'`{k}`' for k in absent)}.")
    out += [
        "- 적법성은 이 시스템이 판정하지 않습니다. 자가 제작 기기를 포함해 **어떤 송신이든 그 전에 KC 적합성평가**가 필요합니다 "
        f"[UNVERIFIED: {CONFORMITY}]" + (f" (`{PROFILE_PREFIX}conformity` = {_value(conformity)})" if conformity is not None else "")
        + ". 전도(conducted) 시험 보드는 더미 로드나 감쇠기에만 연결하고 안테나를 달지 않습니다.",
        "- RF 성능(출력, 편이, 점유 대역폭, 스퓨리어스, 감도, 선택도, ERP 등)은 실험실 측정으로만 확인됩니다(`rf.lab.*`, 증거 없음 → NOT_VERIFIED). "
        "고정구·원리 PASS 는 확인된 모델값 아래의 회로망 판정이며 부품·보드·무전기에 대한 판정이 아닙니다.",
        "",
    ]
    return out


# --------------------------------------------------------------------------- the theory report


def plan_lines(ir: CircuitIR) -> list[str]:
    """The frequency plan with the recorded ``rf.freq_plan`` row statuses (copied) and the rules of each row kind."""
    st = _st()
    rf = rf_of(ir)
    out = ["### 주파수 계획 (`ir.rf.frequency_plan`)", ""]
    lines = list(rf.frequency_plan) if rf is not None else []
    if not lines:
        return out + ["주파수 계획 행이 없습니다.", ""]
    r = ir.validation.latest(FREQ_PLAN_CHECK)
    recorded = {str(row.get("id")): row for row in _rows(r)}
    rows: list[list[object]] = []
    for line in lines:
        f = _num(line.f_hz.value)
        ref = _num(line.ref_hz.value) if line.ref_hz is not None else None
        diff = f"{'+' if f - ref >= 0 else '−'}{_hz(abs(f - ref))}" if f is not None and ref is not None else "-"
        rec = recorded.get(line.id)
        rows.append([f"`{line.id}`", _PLAN_KINDS.get(line.kind, line.kind), _value(line.f_hz), _value(line.ref_hz) if line.ref_hz is not None else "-", diff,
                     _value(line.min_margin_hz) if line.min_margin_hz is not None else "-", _origin(line.f_hz),
                     ", ".join(f"`{x}`" for x in line.points_to) or "-", rec.get("status", "?") if rec is not None else NO_RECORD, line.note or "-"])
    out += [st._table(["행", "종류", "f", "기준 f", "f − 기준 (표시값)", "최소 여유", "f 의 출처", "판정을 가진 곳", "기록된 행 판정", "메모"], rows), ""]
    out += [
        "행의 종류와 `rf.freq_plan` 의 규칙(확인된 주파수의 산술이지 측정이 아님):", "",
        "- 여유(margin): |f − 기준| ≥ 최소 여유이면 PASS, 아니면 FAIL.",
        "- 일치 금지(coincidence): f 와 기준이 같은 수(상대 1e-9)이면 FAIL, 아니면 PASS.",
        "- 응답·게이트(response / gated): 자기 판정이 없고, '판정을 가진 곳' 에 적힌 항목들의 가장 나쁜 상태를 따릅니다. 실험실 항목(`rf.lab.*`)은 증거가 "
        "없어 항상 NOT_VERIFIED 이고, 검사 id 는 현재 설계의 최신 도구 결과를 따릅니다(없거나, 다른 설계의 것이거나, NOT_APPLICABLE 이면 NOT_VERIFIED). "
        "그래서 실험실 항목을 가리키는 행은 측정 전에는 PASS 가 되지 않고, 고정구 행만 가리키는 행은 그 회로망 판정(확인된 모델값 아래, 측정이 아님)을 따릅니다.",
        "- 확인되지 않은(가정) 수를 딛는 행은 NOT_VERIFIED 입니다.", "",
        _result_line(ir, r, FREQ_PLAN_CHECK), "",
    ]
    for row in _rows(r)[:MAX_ROWS]:
        if row.get("status") != ValidationStatus.PASS.value and row.get("reason"):
            out.append(f"  - `{row.get('id')}`: {row.get('status')}: {st._cell(row.get('reason'))}")
    out += ["", *_more(len(_rows(r)))] if _rows(r) else []
    return out


def _profile_statuses(key: str, rows: list[dict]) -> str:
    found = [f"{row.get('status', '?')}" + (f" ({row['check']})" if row.get("check") else "") for row in rows
             if key in [k.strip() for k in str(row.get("key", "")).split(",")]]
    return ", ".join(found) or NO_RECORD


def profile_lines(ir: CircuitIR) -> list[str]:
    """The regulatory profile: every value with its origin, what it is, the document that would ground it, what it is compared with and the recorded row status."""
    st = _st()
    keys = profile_keys_of(ir)
    out = ["### 규제 프로파일 (UNVERIFIED 자리표시값)", ""]
    if not keys:
        return out + ["규제 프로파일이 없습니다 (`ir.rf.profile_keys` 비어 있음).", ""]
    r = ir.validation.latest(PROFILE_CHECK)
    rows_rec = _rows(r)
    rows: list[list[object]] = []
    for k in keys:
        t = ir.parameters.get(k)
        body = _choice_body(t.provenance.note if t is not None else None)
        what = body.split(": ", 1)[1] if body.startswith(f"{k} = ") and ": " in body else body
        what = _UNVERIFIED_RE.sub("", what).rstrip(" -") or NO_RECORD
        suffix = next((s for s in _PROFILE_COMPARES if k.endswith(s)), None)
        rows.append([f"`{k}`", _value(t), _origin(t), what, _grounding_source(t.provenance.note if t is not None else None),
                     _PROFILE_COMPARES.get(suffix or "", "비교 없음 (조건·기록용)"), _profile_statuses(k, rows_rec)])
    out += [st._table(["키", "값", "출처", "무엇", "근거가 될 문서 [UNVERIFIED]", "`rf.regulatory_profile` 이 비교하는 것", "기록된 행 판정"], rows), ""]
    out += [
        "이 값들은 템플릿의 자유 선택으로 `confirm_design` 표에 UNVERIFIED 로 표시되고, 사용자가 표를 확인한 뒤에만 IR 에 들어갑니다(확인 전에는 가정). "
        "어떤 경우에도 `authoritative`(grounding 된 사실)로 기록되지 않습니다: 공식 본문을 `--online` 으로 가져와 보관하고 문장 단위로 대조하기 전까지 모든 수는 자리표시값입니다. "
        "요구사항이 자리표시값 안에 있어도 판정은 NOT_VERIFIED 이며, 넘으면 두 확인된 선택이 서로 어긋난 것(FAIL, 사람이 결정)이지 법적 판정이 아닙니다.", "",
        _result_line(ir, r, PROFILE_CHECK), "",
    ]
    return out


def model_lines(ir: CircuitIR) -> list[str]:
    """The model-grounding list: every ``model.*`` value with its origin, what would ground it and the recorded ``rf.model_grounding`` row (copied)."""
    st = _st()
    keys = model_keys_of(ir)
    rf = rf_of(ir)
    listed = set(rf.model_values) if rf is not None else set()
    out = ["### 모델값 (UNVERIFIED 모델 선택, `rf.model_grounding`)", ""]
    if not keys:
        return out + ["모델값이 없습니다.", ""]
    r = ir.validation.latest(MODEL_CHECK)
    recorded = {str(row.get("key")): row for row in _rows(r)}
    rows: list[list[object]] = []
    for k in keys:
        t = ir.parameters.get(k)
        note = t.provenance.note if t is not None else None
        rec = recorded.get(k)
        value = "모델 카드 (SPICE 텍스트, confirm_design 표의 행)" if t is None and rec is not None and rec.get("card") else _value(t)  # the recorded row says so
        rows.append([f"`{k}`", value, _origin(t), _grounding_source(note), "예" if k in listed else "아니오 (목록에 없음)",
                     (f"{rec.get('status', '?')} — {rec.get('reason', '')}" if rec is not None else NO_RECORD)])
    out += [st._table(["키", "값", "출처", "근거가 될 자료 [UNVERIFIED]", "`ir.rf.model_values` 에 있음", "기록된 행 판정"], rows), ""]
    out += [
        "모델값은 부품의 동작을 대신하는 수(크리스털의 운동 파라미터, 바랙터 C(V), PIN 다이오드 R_on / C_off, 넷리스트에서 빠진 IC 의 포트 저항, 인덕터 Q, "
        "연산 증폭기 GBW, 일반 트랜지스터·다이오드 카드)이며 사실이 아니라 선택입니다. 이 값들로 얻은 고정구·원리 PASS 는 '확인된 모델값 아래의 회로망 판정' 이고 "
        "측정된 부품에 대한 판정이 아닙니다. 데이터시트 사실이나 실험실 기록으로 grounding 되기 전까지 `rf.model_grounding` 은 NOT_VERIFIED 입니다.", "",
        _result_line(ir, r, MODEL_CHECK), "",
    ]
    return out


def _rule(e: Any) -> str:
    unit = _UNITS.get(e.quantity, "")
    if e.bound is not None:
        return bound_text(e.bound)
    if e.tol_abs is not None:
        return f"± {float(e.tol_abs.value):g} {unit}"
    return NO_RECORD


def _ports_text(network: RFNetwork) -> str:
    parts = []
    for p in network.ports:
        z0 = f", {float(p.z0_ohm.value):g} Ω" if p.z0_ohm is not None else ""
        v = f", {float(p.voltage_v.value):g} V" if p.voltage_v is not None else ""
        parts.append(f"`{p.name}` ({p.kind}, 넷 `{p.net}`{z0}{v})")
    return ", ".join(parts)


def _sweep_text(network: RFNetwork) -> str:
    parts = []
    for a in network.sweep:
        params = a.params
        variation = params.get("variation")
        pts, f1, f2 = params.get("points"), params.get("fstart"), params.get("fstop")
        text = f"`{a.id}`: " + " ".join(x for x in (
            str(variation.value) if variation is not None else "",
            f"{pts.value:g}" if pts is not None and _num(pts.value) is not None else "",
            f"{_value(f1)} … {_value(f2)}" if f1 is not None and f2 is not None else "",
        ) if x)
        parts.append(text)
    return "; ".join(parts) or "-"


def _row_where(e: Any) -> str:
    return f"`{e.drive}`" if e.quantity == "s11_db" else f"`{e.drive}` → `{e.to}`"


def fixture_definition_lines(ir: CircuitIR) -> list[str]:
    """Every fixture network: members, ports, states, loss model, sweeps and each row's nominal, rule and the nominal's origin (the formulas first)."""
    st = _st()
    rf = rf_of(ir)
    networks = list(rf.networks) if rf is not None else []
    out = ["### 고정구 회로망 (`spice.rf.*` 가 판정하는 것)", ""]
    if not networks:
        return out + ["고정구 회로망이 없습니다 (이 보드에는 AC 로 판정하는 수동 RF 회로망이 없음).", ""]
    out += [
        "고정구는 회로도에서 잘라 낸 수동 회로망입니다: 덱에는 그 회로망의 부품(members)과 실행기가 더하는 네 가지 소자 - 구동 포트의 ac 1 V 원천과 그 z0 직렬 저항, "
        "다른 포트의 z0 부하(레일·제어 포트는 이상적 DC 원천), `loss_q` 인덕터의 직렬 손실 저항, DC 경로가 없는 노드의 1e12 Ω - 만 있습니다. "
        "ngspice 의 복소 노드 전압에서 전력파 기준으로 계산합니다:", "",
        "      S21 = 2·V_읽기 / V_s · √(R_구동 / R_읽기),   S11 = 2·V_구동 / V_s − 1",
        "      상대 S21(f) = S21_dB(f) − S21_dB(f_기준),   S21 위상 = arg(S21)",
        "      인덕터 손실: R_s = 2π·f_Q·L / Q   (`loss_q`, `q_ref_hz`)", "",
        "프로브(부하 없음)로 가는 'S21' 은 전압비 2·V / V_s 이며 그 위상과 상대 레벨만 의미가 있습니다. 판정하는 행은 그 행 자신의 주파수의 한 점 해석(`ac lin 1 f f`)에서만 읽고 "
        "스위프 점 사이를 보간하지 않습니다. 허용치 행은 |측정 − 공칭| ≤ tol, 한쪽 한계 행은 at_least 이면 측정 ≥ 공칭, at_most 이면 측정 ≤ 공칭(같으면 통과)일 때 PASS 입니다. "
        "|S| 가 정확히 0 이면 −∞ dB 로 읽어 at_most 한계는 통과, at_least·허용치는 실패입니다. 공칭은 템플릿의 계산기(설계한 회로망의 정확한 응답)이거나 확인된 선택값입니다: "
        "그래서 고정구의 PASS 는 컴파일된 넷리스트가 설계한 회로망을 실현했다는 뜻이며, 부품·보드에 대한 판정이 아닙니다.", "",
    ]
    for n in networks:
        out.append(f"#### `{n.id}`" + (f" (블록 `{n.block}`)" if n.block else ""))
        out.append("")
        out.append(f"- 부품: {', '.join(f'`{m}`' for m in n.members)}")
        out.append(f"- 포트: {_ports_text(n)}")
        if n.states:
            out.append("- 상태: " + "; ".join(f"`{s.id}`" + (" (" + ", ".join(f"{p} = {_value(v)}" for p, v in s.port_dc_v.items()) + ")" if s.port_dc_v else "")
                                               + (f", 바인딩 교체 {', '.join(f'`{x}`' for x in s.bindings)}" if s.bindings else "") for s in n.states))
        if n.loss_q:
            out.append(f"- 인덕터 손실 Q (f_Q = {_value(n.q_ref_hz)}): " + ", ".join(f"`{ref}` Q = {float(q.value):g}" for ref, q in n.loss_q.items()))
        if n.bindings:
            out.append(f"- 고정구 바인딩(부품의 SPICE 바인딩 대신): {', '.join(f'`{x}`' for x in n.bindings)}")
        out.append(f"- 스위프: {_sweep_text(n)}")
        out.append("")
        rows = [[f"`{e.id}`", f"`{e.state}`" if e.state else "-", _QUANTITIES.get(e.quantity, e.quantity), _row_where(e), _value(e.at),
                 _value(e.ref_at) if e.ref_at is not None else "-", f"{float(e.nominal.value):.6g} {_UNITS.get(e.quantity, '')}", _rule(e), _origin(e.nominal)]
                for e in n.expectations]
        if rows:
            out += [st._table(["행", "상태", "양", "구동 → 읽기", "f", "기준 f", "공칭", "규칙", "공칭의 출처"], rows), ""]
        else:
            out += ["판정하는 행이 없습니다 (행이 없는 회로망은 아무것도 검증하지 않음).", ""]
        if n.probes:
            out += ["- 기록만 하는 프로브(판정 없음): " + "; ".join(f"`{p.id}` {_QUANTITIES.get(p.quantity, p.quantity)} {_row_where(p)} @ {_value(p.at)}"
                                                           + (f" (기준 {_value(p.ref_at)})" if p.ref_at is not None else "") for p in n.probes), ""]
    return out


def rf_theory_section(ir: CircuitIR) -> list[str]:
    """The theory report's RF section (module docstring); ``[]`` without ``ir.rf``."""
    if rf_of(ir) is None:
        return []
    out = ["## 무선(RF) 설계: 주파수 계획, 규제 프로파일, 모델값, 고정구", "",
           "이 절은 `ir.rf` 와 `ir.parameters` 의 `kr447.*` / `model.*` 값, 그리고 기록된 `rf.*` / `spice.rf.*` 검사를 옮긴 것입니다. 여기의 PASS 는 산술, IR 기하, "
           "또는 확인된 모델값 아래의 회로망·원리에 대한 ngspice 판정이며, IC 가 동작한다는 것도, 무전기의 RF 성능도, 적법성도 뜻하지 않습니다.", ""]
    out += plan_lines(ir)
    out += profile_lines(ir)
    out += model_lines(ir)
    out += fixture_definition_lines(ir)
    return out


# --------------------------------------------------------------------------- the circuit report


def rf_figures_of(ir: CircuitIR, figures: ReportFigures) -> None:
    """The S-parameter charts of every fixture network into its slot ``rf_s21:<network>`` (:mod:`ai_eda.report.rf_figures`); a network without a chart gets the reason."""
    rf = rf_of(ir)
    if rf is None:
        return
    taken = set(figures.figures)
    for n in rf.networks:
        slot = f"{SLOT_RF_S21}{n.id}"
        try:
            drawn, reasons = s_parameter_figures(ir, n, taken=taken)
        except (ValueError, OSError) as e:
            figures.miss(slot, "S-파라미터 그림을 그릴 수 없습니다", str(e))
            continue
        figures.add(slot, *drawn)
        # every reason goes through the ``reason`` argument, which ReportFigures cuts down to basenames (a parser message may name a path)
        if not drawn:
            figures.miss(slot, "S-파라미터 그림이 없습니다", "; ".join(reasons) or "그릴 곡선이 없습니다")
        elif reasons:
            figures.miss(slot, "일부 곡선을 그리지 않았습니다", "; ".join(reasons))


def _keepout_area(k: Any) -> str:
    if k.rect is not None:
        x, y, w, h = (float(v) for v in k.rect.value)
        return f"x {x:g}–{x + w:g}, y {y:g}–{y + h:g} mm"
    pts = k.polygon.value if k.polygon is not None else []
    return f"다각형 {len(pts)}점"


def block_lines(ir: CircuitIR) -> list[str]:
    """The RF blocks (regions, chain, shield can, interface ports) with the ``block.interface.*`` results, the keep-outs (whether each clips the
    plane zones), and ``pcb.keepout`` - which also judges the block regions and can fences, so it is copied whenever a block has a region or a can."""
    st = _st()
    rf = rf_of(ir)
    blocks = list(rf.blocks) if rf is not None else []
    keepouts = list(ir.pcb.keepouts) if ir.pcb is not None else []
    out = ["### RF 블록과 keep-out", ""]
    if not blocks and not keepouts:
        return out + ["RF 블록도 keep-out 도 없습니다.", ""]
    if blocks:
        rows: list[list[object]] = []
        for b in blocks:
            region = "-"
            if b.region is not None:
                x0, y0, x1, y1 = b.region.box()
                region = f"x {x0:g}–{x1:g}, y {y0:g}–{y1:g} mm"
            rows.append([f"`{b.id}`", b.title or "-", len(b.refs), " → ".join(b.chain) or "-", f"`{b.shield_ref}`" if b.shield_ref else "-", region,
                         ", ".join(f"`{p.name}` ({p.kind}, `{p.net}`)" for p in b.ports) or "-"])
        out += [st._table(["블록", "제목", "부품 수", "신호 순서", "차폐 캔", "배치 영역", "인터페이스 포트"], rows), ""]
        out += ["배치 영역과 신호 순서는 `placement.rf_floorplan` 이 따르는 템플릿의 선택값입니다(영역보다 큰 부품이 있으면 겹치지 않고 거부). "
                "블록 사이의 포트는 `block.interface.<넷>` 이 종류·기준 넷·임피던스·주파수·전압이 서로 맞는지 IR 산술로 봅니다:", ""]
        results = _latest(ir, ("block.interface.",))
        out += [_result_line(ir, r, r.check_id) for r in results] or [f"- `block.interface.*`: {NO_RECORD}"]
        out.append("")
    if keepouts:
        rows = [[f"`{k.id}`", ", ".join(k.layers), _keepout_area(k), ", ".join(k.forbids), ", ".join(f"`{x}`" for x in k.allowed_refs) or "-",
                 ", ".join(f"`{x}`" for x in k.allowed_nets) or "-", _plane_cell(k), k.reason] for k in keepouts]
        out += [st._table(["keep-out", "층", "영역", "금지", "예외 부품", "예외 넷", "평면 존", "이유"], rows), "",
                "keep-out 은 자기가 금지하는 항목(트랙·비아 등)에 대해 라우터의 장애물이고, 컴파일된 보드에 규칙 영역으로 쓰입니다. 평면 존은 `zones` 를 금지하는 "
                "keep-out 만 자릅니다(그 층에서, 예외 넷이 아닌 평면만; 다각형은 외접 사각형으로): '평면 존' 열은 이 규칙을 keep-out 마다 적은 것입니다. "
                "예외(허용 부품·넷)는 IR 검사 `pcb.keepout` 만 알며, "
                "KiCad 의 규칙 영역에는 예외가 없어 허용된 것의 통로를 잘라 낸 모양으로 씁니다(KiCad 가 읽는 것은 측정되지 않음).", ""]
    if keepouts or any(b.region is not None or b.shield_ref for b in blocks):
        # the check judges more than the keep-outs: every block part inside its region and every part under a can inside the fence
        out += ["`pcb.keepout` (IR 기하 검사, DRC 아님)은 keep-out 이 금지하는 것이 그 안에 없는지와 함께, 블록의 부품이 배치 영역 안에, 차폐 캔 밑의 "
                "부품이 캔의 울타리 안에 있는지를 봅니다:", "",
                _result_line(ir, ir.validation.latest(KEEPOUT_CHECK), KEEPOUT_CHECK), ""]
    return out


def _plane_cell(k: Any) -> str:
    """Whether a keep-out clips the stackup's plane zones (``agents.pcb._clip_planes``' rule, read from the keep-out's own fields, never its reason)."""
    if not forbids(k, "zones"):
        return "유지 (`zones` 를 금지하지 않음)"
    kept = list(allowed_nets(k))
    return "자름" + (f" (예외 넷 {', '.join(f'`{n}`' for n in kept)} 의 평면은 유지)" if kept else "")


def _row_cells(e: Any, r: ValidationResult | None) -> list[object]:
    unit = _UNITS.get(e.quantity, "")
    d = r.details if r is not None and isinstance(r.details, dict) else {}
    nominal = _num(d.get("nominal"))
    if nominal is None:
        nominal = _num(e.nominal.value)
    zero = d.get("zero_magnitude")
    if r is None:
        measured = NO_RECORD
    elif isinstance(zero, dict) and "at" in zero and d.get("measured") is None:  # an exact zero |S| at the row's own frequency (JSON has no -inf)
        measured = f"−∞ {unit} (|S| = 0)"
    else:
        measured = _level(d.get("measured"), unit)
    deviation = _level(d.get("deviation"), unit) if r is not None and "deviation" in d else "-"
    return [f"`{e.id}`", f"`{e.state}`" if e.state else "-", _QUANTITIES.get(e.quantity, e.quantity), _row_where(e), _value(e.at),
            f"{nominal:.6g} {unit}" if nominal is not None else NO_RECORD, _rule(e), measured, deviation, str(r.status) if r is not None else NO_RECORD]


def _deck_lines(summary: ValidationResult) -> list[str]:
    decks = summary.details.get("decks") if isinstance(summary.details, dict) else None
    out: list[str] = []
    for d in decks if isinstance(decks, list) else []:
        if not isinstance(d, dict):
            continue
        dc = d.get("dc_path") if isinstance(d.get("dc_path"), dict) else {}
        extra = [f"추가 소자 {len(d.get('added') or [])}개", f"DC 경로 저항 {len(dc.get('resistors') or [])}개"]
        if d.get("open_nets"):
            extra.append(f"고정구에서 열린 넷 {', '.join(f'`{x}`' for x in d['open_nets'])}")
        if d.get("problem"):
            extra.append(f"문제: {_st()._cell(d['problem'])}")
        state = f"상태 `{d.get('state')}`, " if d.get("state") else ""
        out.append(f"  - 덱 `{d.get('stem')}` ({state}구동 `{d.get('drive')}`): " + ", ".join(extra))
    names = _evidence_names(summary)
    if names:
        out.append(f"  - 증거 파일 (덱 + rawfile, 해시와 함께 기록, {len(names)}개): {', '.join(f'`{x}`' for x in names)}")
    return out


def _evidence_names(summary: ValidationResult) -> list[str]:
    """Every recorded evidence file once, named by its path under the fixture folder (the deck's own folder): each analysis writes
    ``<stem>/<analysis>/<stem>.ac.raw``, so a basename alone would fold them into one. Never an absolute path; outside that folder a basename."""
    paths = [e.path.replace("\\", "/") for e in summary.evidence if e.path]
    roots = sorted({p.rsplit("/", 1)[0] for p in paths if p.endswith(".cir") and "/" in p})
    out: list[str] = []
    for p in paths:
        root = next((r for r in roots if p.startswith(r + "/")), None)
        name = p[len(root) + 1:] if root is not None else p.rsplit("/", 1)[-1]
        if name not in out:
            out.append(name)
    return sorted(out)


def _probe_lines(summary: ValidationResult | None) -> list[str]:
    probes = summary.details.get("probes") if summary is not None and isinstance(summary.details, dict) else None
    if not isinstance(probes, dict) or not probes:
        return []
    out = ["- 기록된 프로브 (판정 없음):"]
    for key, row in probes.items():
        if isinstance(row, dict):
            out.append(f"  - `{key}`: {_st()._cell(row.get('note', ''))}")
    return out


def fixture_result_lines(ir: CircuitIR, figures: ReportFigures) -> list[str]:
    """Per fixture network: the recorded summary, every row's recorded measurement and status, probes, decks, evidence and the S-parameter charts."""
    st = _st()
    rf = rf_of(ir)
    networks = list(rf.networks) if rf is not None else []
    out = ["### 고정구 검증 (`spice.rf.*`, ngspice)", ""]
    if not networks:
        return out + ["고정구 회로망이 없습니다.", ""]
    out += ["각 회로망의 요약(`spice.rf.<회로망>`)과 행(`spice.rf.<회로망>[.<상태>].<행>`)의 기록을 옮긴 것입니다. 측정값·편차·판정은 기록 그대로이며, "
            "PASS 는 '확인된 모델값 아래의 회로망 판정(측정된 부품 아님; 회로도 수준 - 트랙·비아·접지 귀로 인덕턴스 없음)' 입니다. 덱과 rawfile 은 해시와 함께 증거로 "
            "기록되며, rawfile 에는 ngspice 의 Date 줄이 있어 결정론적 산출물이 아니라 증거입니다.", ""]
    for n in networks:
        summary = fixture_summary(ir, n.id)
        out += [f"#### `{n.id}`" + (f" (블록 `{n.block}`)" if n.block else ""), "", _result_line(ir, summary, f"spice.rf.{n.id}"), ""]
        rows = []
        for e in n.expectations:
            sid = f".{e.state}" if e.state else ""
            rows.append(_row_cells(e, ir.validation.latest(f"spice.rf.{n.id}{sid}.{e.id}")))
        if rows:
            out += [st._table(["행", "상태", "양", "구동 → 읽기", "f", "공칭 (기록)", "규칙", "측정 (기록)", "편차 (기록)", "판정 (기록)"], rows), ""]
        not_pass = []
        for e in n.expectations:
            sid = f".{e.state}" if e.state else ""
            r = ir.validation.latest(f"spice.rf.{n.id}{sid}.{e.id}")
            if r is not None and r.status is not ValidationStatus.PASS:
                not_pass.append(f"  - `{r.check_id}`: {r.status} — {_message(r.message)}")
        if not_pass:
            out += ["- PASS 가 아닌 행의 기록된 메시지:", *not_pass, ""]
        probe = _probe_lines(summary)
        if probe:
            out += probe + [""]
        if summary is not None:
            decks = _deck_lines(summary)
            if decks:
                out += ["- 덱 (기록 그대로):", *decks, ""]
        out += figures.lines(f"{SLOT_RF_S21}{n.id}")
    return out


def deviation_lines(ir: CircuitIR) -> list[str]:
    """The ``rf.deviation`` chain as recorded: the result line and each factor's value, source and status (copied)."""
    st = _st()
    r = ir.validation.latest(DEVIATION_CHECK)
    if r is None:
        return []
    out = ["### 주파수 편이 사슬 (`rf.deviation`)", "",
           "      Δf = N · K_pm · a · V_max / (2π · τ_i)", "",
           "N 은 체배수, K_pm 은 위상 변조 탱크 고정구(`pm_mod*`)의 bias_lo / bias_hi 위상 차를 전압 차로 나눈 기울기의 합, a 는 오디오 경로 결합(`spice.pm_couple_*` 중 가장 큰 것), "
           "V_max 는 스플래터 필터 출력의 최댓값(`spice.pm_drive_peak`), τ_i 는 적분기 시정수(`spice.integrator_1k` 가 확인)입니다. 모든 인자가 이 설계에서 PASS 일 때만 "
           "PASS / FAIL 이 되며 '확인된 모델값 아래' 의 원리 판정입니다. 실제 편이는 실험실 항목입니다.", "",
           _result_line(ir, r, DEVIATION_CHECK), ""]
    factors = r.details.get("factors") if isinstance(r.details, dict) else None
    rows = [[f.get("factor", "?"), f"{_num(f.get('value')):.6g}" if _num(f.get("value")) is not None else NO_RECORD, f.get("unit") or "-", f.get("source") or "-",
             f.get("status", "?"), f.get("reason") or "-"] for f in factors or [] if isinstance(f, dict)]
    if rows:
        out += [st._table(["인자", "값 (기록)", "단위", "출처 (기록)", "인자 판정 (기록)", "이유"], rows), ""]
    return out


def rf_circuit_section(ir: CircuitIR, figures: ReportFigures) -> list[str]:
    """The circuit report's RF section (module docstring); ``[]`` without ``ir.rf``."""
    if rf_of(ir) is None:
        return []
    out = ["## 무선(RF) 블록과 고정구 검증", ""]
    out += block_lines(ir)
    out += fixture_result_lines(ir, figures)
    out += deviation_lines(ir)
    return out


# --------------------------------------------------------------------------- the final report


def summary_lines(ir: CircuitIR) -> list[str]:
    """The latest RF results (fixture summaries, ``rf.*`` except the lab items, ``block.interface.*``, ``power.*``, ``pcb.keepout``) in one table, copied."""
    st = _st()
    rf = rf_of(ir)
    design_hash = ir.content_hash()
    ids = [f"spice.rf.{n.id}" for n in (rf.networks if rf is not None else [])]
    results = [r for r in (ir.validation.latest(i) for i in ids) if r is not None]
    results += [r for r in _latest(ir, RF_CHECK_PREFIXES) if not r.check_id.startswith(LAB_PREFIX)]
    keep = ir.validation.latest(KEEPOUT_CHECK)
    if keep is not None:
        results.append(keep)
    out = ["### RF 검사의 최신 결과 (저장된 상태 그대로)", ""]
    if not results:
        return out + [f"RF 검사 결과 {NO_RECORD}.", ""]
    rows = [[f"`{r.check_id}`", str(r.status), st._tool_text(r.tool, r.tool_version), st._freshness(r, design_hash), _message(r.message)] for r in results]
    out += [st._table(["검사", "판정", "도구", "대상 IR", "메시지"], rows), ""]
    return out


def lab_lines(ir: CircuitIR) -> list[str]:
    """The lab-only list (``ir.rf.lab_items``) with each ``rf.lab.<id>`` status copied."""
    st = _st()
    rf = rf_of(ir)
    items = list(rf.lab_items) if rf is not None else []
    out = ["### 실험실에서만 확인되는 항목 (`rf.lab.*`)", ""]
    if not items:
        return out + ["실험실 항목이 없습니다.", ""]
    rows = []
    for x in items:
        r = ir.validation.latest(f"{LAB_PREFIX}{x.id}")
        rows.append([f"`{LAB_PREFIX}{x.id}`", f"`{x.block}`" if x.block else "-", x.what, ", ".join(x.instruments) or "-", x.reason, str(r.status) if r is not None else NO_RECORD])
    out += [st._table(["항목", "블록", "무엇", "장비", "이유", "기록된 판정"], rows), "",
            "실험실 증거를 가져오는 도구는 이 버전에 없으므로 모든 항목은 NOT_VERIFIED 'no lab evidence' 이며, 어떤 설계 검사도 이 항목을 대신하지 않습니다. "
            "송신이 들어가는 측정(출력, 편이, 점유 대역폭, 스퓨리어스, ERP 등)은 KC 적합성평가와 그 시험 조건을 확인한 뒤 전도 측정(더미 로드·감쇠기)으로만 합니다 [UNVERIFIED].", ""]
    return out


def _ma(t: Traced) -> str:
    """A current in mA (one unit across a rail row: 86 mA - 161 mA, never 86 mA - 0.161 A); another unit as :func:`_value` prints it."""
    if t.unit != "A" or _num(t.value) is None:
        return _value(t)
    return f"{float(t.value) * 1e3:.6g} mA"


def rail_lines(ir: CircuitIR) -> list[str]:
    """The rail budgets (``ir.rf.rails``) with the ``power.rail_budget.*`` / ``power.headroom.*`` statuses copied."""
    st = _st()
    rf = rf_of(ir)
    rails = list(rf.rails) if rf is not None else []
    if not rails:
        return []
    rows = []
    for x in rails:
        budget = ir.validation.latest(f"power.rail_budget.{x.rail}")
        head = ir.validation.latest(f"power.headroom.{x.regulator_ref}")
        rows.append([f"`{x.rail}`", f"`{x.regulator_ref}`", _value(x.v_out), f"{_ma(x.i_min)} – {_ma(x.i_max)}", _ma(x.i_rating) if x.i_rating is not None else "-",
                     _value(x.dropout_v) if x.dropout_v is not None else "-", _value(x.path_r_ohm) if x.path_r_ohm is not None else "-", _origin(x.i_max),
                     str(budget.status) if budget is not None else NO_RECORD, str(head.status) if head is not None else NO_RECORD])
    return ["### 전원 레일 예산 (`ir.rf.rails`)", "",
            st._table(["레일", "레귤레이터", "V_out", "부하 전류", "정격", "드롭아웃", "경로 저항", "전류의 출처", "`power.rail_budget` (기록)", "`power.headroom` (기록)"], rows), "",
            "부하 전류·정격·드롭아웃은 데이터시트 사실이라 grounding 되기 전에는 선택값이고, 이 행들은 확인된 값끼리 어긋날 때만 FAIL, grounding 된 값일 때만 PASS 이며 "
            "그 밖에는 NOT_VERIFIED 입니다.", ""]


def limit_lines() -> list[str]:
    return [
        "### 정직한 한계", "",
        "- 여기의 PASS 는 산술, IR 기하, 또는 확인된 모델값 아래의 회로망·원리에 대한 ngspice 판정입니다. IC 가 동작한다는 뜻도, RF 성능이나 적법성이 확인되었다는 뜻도 아닙니다.",
        "- 모든 IC(FM IF, 믹서, 증폭기, PA, TCXO, 레귤레이터, 비교기, 로직)는 넷리스트에서 빠져 있고, 트랜지스터·다이오드·바랙터·크리스털·연산 증폭기는 일반 카드나 확인된 모델값으로 "
        "시뮬레이션됩니다(`rf.model_grounding`).",
        "- 잡음 지수, S-파라미터 파일(Touchstone), PA 의 대신호 동작은 모델이 없어 다루지 않고(`noise` / `sp` 금지), UHF 반송파를 오디오 시간 동안 과도 해석하지도 않습니다.",
        "- 고정구는 회로도 수준입니다: 트랙·비아·접지 귀로 인덕턴스가 없으므로 보드 위의 응답, 특히 저지대역은 다를 수 있습니다. 배치·배선의 기하 검사는 RF 품질을 말하지 않습니다.",
        "- 규제 수치는 검증되지 않은 자리표시값입니다. KC 적합성평가 전에는 어떤 송신도 하지 않습니다(자가 제작 포함) [UNVERIFIED].",
        "",
    ]


def rf_final_section(ir: CircuitIR) -> list[str]:
    """The final report's RF summary (module docstring); ``[]`` without ``ir.rf``."""
    if rf_of(ir) is None:
        return []
    out = ["## 무선(RF) 검증 요약과 실험실 항목", ""]
    out += summary_lines(ir)
    out += lab_lines(ir)
    out += rail_lines(ir)
    out += limit_lines()
    return out


__all__ = [
    "BANNER_TITLE",
    "RF_CHECK_PREFIXES",
    "SLOT_RF_S21",
    "block_lines",
    "deviation_lines",
    "fixture_definition_lines",
    "fixture_result_lines",
    "lab_lines",
    "model_keys_of",
    "model_lines",
    "plan_lines",
    "profile_banner",
    "profile_dependents",
    "profile_keys_of",
    "profile_lines",
    "rail_lines",
    "rf_circuit_section",
    "rf_figures_of",
    "rf_final_section",
    "rf_of",
    "rf_theory_section",
    "summary_lines",
]
