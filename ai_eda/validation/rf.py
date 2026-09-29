"""RF design checks of ``ir.rf``: IR arithmetic, the recorded SPICE verdicts and the registered calculators - never a measurement.

Invariant: a verdict here is a deterministic comparison of numbers the IR
states (``ir.rf``, ``ir.parameters``, the confirmed requirements read by
:func:`ai_eda.design.inputs.read_inputs`), numbers a registered calculator
computes from them (``calc.power.rail_budget`` / ``calc.regulator.headroom``,
called through :data:`ai_eda.tools.calc.recompute.CALCULATORS` by role) and
verdicts other tools already recorded (``spice.<id>`` of the design deck,
``spice.rf.<network>[.<state>].<exp>`` of the fixture runner) - read by check
id, used only when they are about the current design (stamped with its
design hash, tool-backed) and, where a *measured number* is taken from them,
only while every evidence file they name is on disk with its recorded hash.
Every derived number stays in ``details``; nothing is written into the IR.
A PASS here is arithmetic, or a verdict about a network or a principle under
confirmed model values: it never says that an IC works, that the radio meets
its RF performance or that it is legal. What cannot be judged is
NOT_VERIFIED naming why; ``rf.regulatory_profile`` and ``rf.lab.<id>`` can
never PASS.

"Confirmed" means the value, or every input a derived value was computed
from (followed through ``Provenance.derived_from``), is the user's
(``user_requirement``: a typed answer or a confirmed template choice) or
authoritative. "Grounded" means ``authoritative`` with a source document -
a confirmed choice is confirmed, not grounded. A verdict that rests on an
unconfirmed value (``assumption`` / ``llm_generated``, or an input nobody can
find) is NOT_VERIFIED naming it, never PASS or FAIL.

The validator (:class:`RFChecksValidator`, tool :data:`RF_TOOL`) runs on a
design with ``ir.rf`` and again after the SPICE stage (it consumes
``spice``). Results:

* ``rf.freq_plan`` - one row per :class:`~ai_eda.ir.rf.PlanLine`: a
  ``margin`` row PASSes when ``|f - ref| >= min_margin`` and FAILs otherwise;
  a ``coincidence`` row names a pair that must not coincide and FAILs when
  ``f`` equals ``ref`` (to 1e-9 relative - the same number), PASS otherwise;
  a ``response`` / ``gated`` row has no verdict of its own and takes the
  worst of what ``points_to`` names: an ``rf.lab.<id>`` is NOT_VERIFIED (no
  lab evidence), a check id the latest recorded result of that check when it
  is tool-backed and about this design (a stale, missing, tool-less or
  NOT_APPLICABLE one is NOT_VERIFIED). Only with a frequency plan.
* ``rf.regulatory_profile`` - the requirements against the profile values
  ``ir.rf.profile_keys`` names, matched by suffix: ``.max_power`` (tx_power
  at most), ``.max_deviation`` (frequency_deviation at most), ``.max_obw``
  (occupied_bandwidth at most), ``.freq_tolerance`` (frequency_tolerance at
  most), ``.band_low`` / ``.band_high`` / ``.channel_raster`` (the
  carrier_frequency inside the band, on the raster counted from
  ``band_low``: the band edges are the first and last channel centres). An
  exceedance of a confirmed profile value is FAIL (``repair: human``: two
  confirmed choices contradict each other - not a legal verdict); anything
  within is NOT_VERIFIED, because the profile is an unverified user-confirmed
  placeholder, never a grounded fact; a stated ``erp`` / ``eirp`` is named as
  not compared (the profile's reference point is an assumption). Never PASS.
* ``rf.model_grounding`` - every ``model.*`` value (``ir.rf.model_values``
  and every ``model.*`` key of ``ir.parameters``): PASS only when every one is
  grounded, else NOT_VERIFIED naming them - a fixture or principle PASS
  resting on them is a verdict under confirmed model values, not about a
  measured part.
* ``rf.deviation`` - Delta_f = N K_pm a V_max / (2 pi tau_i) from
  ``rf.n_mult``, ``tx.tau_i``, K_pm = the sum over the ``pm_mod*`` fixture
  networks of each tank's chord slope (phi_hi - phi_lo) / (V_hi - V_lo) from
  the ``bias_lo`` / ``bias_hi`` phase results and the states' ``port_dc_v``,
  a = the largest of ``spice.pm_couple_300`` / ``_1k`` / ``_3k``,
  V_max = ``spice.pm_drive_peak`` (tau_i checked by ``spice.integrator_1k``).
  PASS against the confirmed ``frequency_deviation`` only when every factor's
  check PASSed on this design ("under confirmed model values"), FAIL when
  those PASSing factors multiply to more; NOT_VERIFIED otherwise - always on
  a board whose K_pm is the model constant ``model.k_pm``, and when the tanks'
  phase does not move with the bias (K_pm 0: no deviation at all). The linearity
  figure (phi_hi + phi_lo - 2 phi_nom) / (phi_hi - phi_lo) of each tank is
  recorded without a verdict. The real deviation is a lab item.
* ``rf.lab.<id>`` - one per :class:`~ai_eda.ir.rf.LabItem`: NOT_VERIFIED
  "no lab evidence" (no lab-evidence importer exists in this version).
* ``block.interface.<net>`` - the ports of two or more blocks on one net
  (the interface nets keep their names in a composition): the same kind and
  reference net, equal ``z0_ohm`` (ports), equal ``frequency_hz`` where
  stated, equal ``voltage_v`` (rails / controls), at most one ``out``. PASS
  when everything compared agrees (IR arithmetic), FAIL on a disagreement,
  NOT_VERIFIED when a side does not state what the other does or no port
  drives the net.
* ``power.rail_budget.<rail>`` - I_rating - I_max (``calc.power.rail_budget``);
  ``power.headroom.<regulator_ref>`` - V_in,min - I_max R_path - V_dropout -
  V_out (``calc.regulator.headroom``) at V_in,min = the confirmed pack
  cut-off ``power.pack_cutoff_v``, or - for a regulator whose pins touch
  another rail of ``ir.rf.rails`` - that rail's ``v_out`` (it is fed from
  it). A negative margin / headroom from confirmed values is FAIL (an unstated
  dropout or path resistance counts as 0, which only raises the headroom, so
  a negative bound is still a contradiction); PASS only when the currents,
  the rating, the dropout and the path resistance are grounded; every other
  case NOT_VERIFIED ("ungrounded").

A check id this validator produced earlier that the current IR no longer
produces (a removed lab item, block port, rail, plan or profile; or no
``ir.rf`` at all) gets a superseding NOT_APPLICABLE result, so an old verdict
never stays the latest word.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ai_eda.ir import CircuitIR, Evidence, ProvenanceKind, Traced, ValidationResult, ValidationStatus, assumption, worst_status
from ai_eda.tools.calc.quantity import parse_unit
from ai_eda.tools.calc.recompute import CALCULATORS, resolve_input
from ai_eda.tools.spice.stage import CHECK_ID as SPICE_CHECK_ID
from ai_eda.validation.base import ValidationContext, Validator
from ai_eda.validation.registry import default_registry

S = ValidationStatus

#: the tool every result here names (a deterministic comparison, see the module docstring)
RF_TOOL = "rf.checks"
#: bumped when a judgement rule changes
RF_VERSION = "0.1"
#: what the results are: arithmetic on IR numbers (and, for response / gated plan rows, the recorded verdicts they point to)
KIND = "ir_arithmetic"
#: what rf.deviation is: recorded SPICE verdicts (their measured numbers) multiplied by arithmetic
DEVIATION_KIND = "recorded_spice_verdicts+arithmetic"
FREQ_PLAN_CHECK = "rf.freq_plan"
PROFILE_CHECK = "rf.regulatory_profile"
MODEL_CHECK = "rf.model_grounding"
DEVIATION_CHECK = "rf.deviation"
LAB_PREFIX = "rf.lab"
INTERFACE_PREFIX = "block.interface"
RAIL_BUDGET_PREFIX = "power.rail_budget"
HEADROOM_PREFIX = "power.headroom"
#: every check id this validator produces starts with one of these (and it retires what it no longer produces)
OWNED_PREFIXES: tuple[str, ...] = ("rf.", f"{INTERFACE_PREFIX}.", f"{RAIL_BUDGET_PREFIX}.", f"{HEADROOM_PREFIX}.")
#: the fixture runner's check ids (``spice.rf.<network>[.<state>].<exp>``; ``ai_eda.tools.spice.stage.RF_CHECK_PREFIX``)
FIXTURE_PREFIX = "spice.rf"
#: the prefix of a modelling number no datasheet or measurement grounds
MODEL_PREFIX = "model."
#: the phase-modulator fixture networks and their three bias states
PM_NETWORK_PREFIX = "pm_mod"
PM_LO, PM_NOM, PM_HI = "bias_lo", "bias_nom", "bias_hi"
#: the design parameters of the deviation chain
N_MULT_KEY = "rf.n_mult"
TAU_I_KEY = "tx.tau_i"
MODEL_K_PM_KEY = "model.k_pm"
#: the design-deck expectations of the deviation chain (``spice.<id>``)
PM_COUPLE_CHECKS: tuple[str, ...] = (f"{SPICE_CHECK_ID}.pm_couple_300", f"{SPICE_CHECK_ID}.pm_couple_1k", f"{SPICE_CHECK_ID}.pm_couple_3k")
PM_DRIVE_CHECK = f"{SPICE_CHECK_ID}.pm_drive_peak"
INTEGRATOR_CHECK = f"{SPICE_CHECK_ID}.integrator_1k"
DEVIATION_KEY = "frequency_deviation"
#: the confirmed pack cut-off: the minimum input of a regulator fed from the pack
PACK_CUTOFF_KEY = "power.pack_cutoff_v"
RAIL_BUDGET_CALC = "calc.power.rail_budget"
HEADROOM_CALC = "calc.regulator.headroom"
#: profile key suffix -> (the requirement it bounds from above, the unit both carry)
PROFILE_LIMITS: dict[str, tuple[str, str]] = {
    ".max_power": ("tx_power", "W"),
    ".max_deviation": ("frequency_deviation", "Hz"),
    ".max_obw": ("occupied_bandwidth", "Hz"),
    ".freq_tolerance": ("frequency_tolerance", "ppm"),
}
BAND_LOW, BAND_HIGH, RASTER = ".band_low", ".band_high", ".channel_raster"
CARRIER_KEY = "carrier_frequency"
#: the radiated-power keys a conducted power limit is not compared with
RADIATED_KEYS: tuple[str, ...] = ("erp", "eirp")
#: why the profile check never PASSes
UNVERIFIED_PROFILE = ("the regulatory profile is an UNVERIFIED user-confirmed placeholder (no official text archived and grounded), "
                      "so it is never evidence of legality and this check never PASSes")
#: how a PASS resting on model values is worded
UNDER_MODEL_VALUES = "under confirmed model values (a network / principle verdict, not a measured part)"
NO_LAB_EVIDENCE = "no lab evidence"
#: "the same number" for two frequencies / impedances / voltages computed from the same inputs
REL_TOL = 1e-9
#: a carrier is on the raster when its channel index is an integer to this
RASTER_TOL = 1e-6
#: units a dimensionless number may carry
_DIMENSIONLESS = (None, "", "ratio")


def _rf(ir: CircuitIR) -> Any:
    """``ir.rf`` (``None`` when the design states no RF content, or when this build's IR has no such field)."""
    return getattr(ir, "rf", None)


def _result(check_id: str, status: S, message: str, evidence: list[Evidence] | None = None, **details: Any) -> ValidationResult:
    return ValidationResult(check_id=check_id, status=status, message=message, tool=RF_TOOL, tool_version=RF_VERSION,
                            evidence=list(evidence or []), details={"kind": KIND, **details})


def _worst(rows: list[dict]) -> S:
    return worst_status(S(r["status"]) for r in rows) if rows else S.NOT_APPLICABLE


def _hz(f: float) -> str:
    """``447.5625 MHz``, ``21.4 MHz``, ``450 kHz`` - exact enough to tell channels apart."""
    for scale, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if abs(f) >= scale:
            return f"{f / scale:.10g} {unit}"
    return f"{f:.10g} Hz"


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b))


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def value_in(t: Traced | None, unit: str | None, what: str) -> tuple[float | None, str | None]:
    """``(number in unit, None)`` of a traced number (an SI prefix on its unit is applied), or ``(None, why)``.

    ``unit`` ``None`` accepts a dimensionless number (no unit, ``""`` or ``ratio``).
    """
    if t is None:
        return None, f"{what}: not stated"
    v = t.value
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None, f"{what}: {v!r} is not a finite number"
    if unit is None:
        if t.unit in _DIMENSIONLESS:
            return float(v), None
        return None, f"{what}: carries unit {t.unit!r}, a dimensionless number was expected"
    if t.unit == unit:
        return float(v), None
    parsed = parse_unit(t.unit) if t.unit else None
    if parsed is not None and parsed[0] == unit:
        return float(v) * 10.0 ** parsed[1], None
    return None, f"{what}: carries unit {t.unit!r}, not {unit}"


# --------------------------------------------------------------------------- what a number rests on


@dataclass
class Basis:
    """What a traced number rests on (module docstring): confirmed / grounded, and the values that make it not so."""

    weak: list[str] = field(default_factory=list)  # not confirmed: assumption / llm_generated / an input nobody can find
    ungrounded: list[str] = field(default_factory=list)  # confirmed, but not a grounded fact

    @property
    def confirmed(self) -> bool:
        return not self.weak

    @property
    def grounded(self) -> bool:
        return not self.weak and not self.ungrounded

    def add(self, other: Basis) -> Basis:
        self.weak += [w for w in other.weak if w not in self.weak]
        self.ungrounded += [u for u in other.ungrounded if u not in self.ungrounded]
        return self


def _lookup(ir: CircuitIR, key: str) -> Traced | None:
    """The traced value an input id names: ``ir.parameters`` / ``ir.si`` / the stackup / a requirement key (the recompute's
    resolution), else a requirement id, else an ``rf.*`` path."""
    t = resolve_input(ir, key)
    if t is not None:
        return t
    req = next((r for r in ir.requirements.requirements if r.id == key), None)
    if req is not None:
        return req.value
    rf = _rf(ir)
    lookup = getattr(rf, "lookup", None)
    return lookup(key) if callable(lookup) else None


def basis(ir: CircuitIR, name: str, t: Traced | None) -> Basis:
    """What ``t`` (called ``name``) rests on, following a derived value's inputs to their sources."""
    out = Basis()
    if t is not None:
        _walk(ir, name, t, out, set())
    return out


def _walk(ir: CircuitIR, name: str, t: Traced, out: Basis, seen: set[str]) -> None:
    p = t.provenance
    if p.kind is ProvenanceKind.DERIVED:
        if not p.derived_from:
            out.weak.append(f"{name} (derived, but its inputs are not recorded)")
            return
        for key in p.derived_from:
            if key in seen:
                continue
            seen.add(key)
            src = _lookup(ir, key)
            if src is None:
                out.weak.append(f"{key} (an input of {name} that nothing in the IR names)")
            else:
                _walk(ir, key, src, out, seen)
        return
    if p.kind is ProvenanceKind.AUTHORITATIVE and p.source is not None:
        return
    if p.kind in (ProvenanceKind.USER_REQUIREMENT, ProvenanceKind.AUTHORITATIVE):
        out.ungrounded.append(f"{name} ({p.kind.value}{', no source document' if p.kind is ProvenanceKind.AUTHORITATIVE else ''})")
        return
    out.weak.append(f"{name} ({p.kind.value})")


def _bases(ir: CircuitIR, named: Iterable[tuple[str, Traced | None]]) -> Basis:
    out = Basis()
    for name, t in named:
        out.add(basis(ir, name, t))
    return out


# --------------------------------------------------------------------------- reading recorded results


@dataclass
class _Ctx:
    """One validation pass: the IR and its design hash (computed once)."""

    ir: CircuitIR
    current: str

    def result(self, check_id: str, *, numbers: bool = False) -> tuple[ValidationResult | None, str | None]:
        """The latest ``check_id`` result when it is about this design, else ``(None, why)``.

        About this design: tool-backed and stamped with the current design
        hash. ``numbers``: a measured number will be taken from it, so every
        evidence file it names with a hash must be on disk with that hash.
        """
        r = self.ir.validation.latest(check_id)
        if r is None:
            return None, f"{check_id}: no result recorded"
        if not r.is_tool_backed:
            return None, f"{check_id}: {r.status.value} without a tool (an opinion, not evidence)"
        if r.ir_hash != self.current:
            return None, f"{check_id}: judged on another design version (not about this IR)"
        if numbers:
            files = [e for e in r.evidence if e.path and e.content_hash]
            if not files:
                return None, f"{check_id}: no evidence file with a recorded hash"
            for e in files:
                path = Path(e.path)  # type: ignore[arg-type]
                if not path.is_file():
                    return None, f"{check_id}: evidence {path.name} is not on disk"
                if _sha256(path) != e.content_hash:
                    return None, f"{check_id}: evidence {path.name} does not match its recorded hash"
        return r, None

    def passed_number(self, check_id: str, unit: str | None) -> tuple[float | None, ValidationResult | None, str | None]:
        """``(measured, result, None)`` of a PASSing current result with a finite ``details["measured"]``, else ``(None, result?, why)``."""
        r, why = self.result(check_id, numbers=True)
        if r is None:
            return None, None, why
        if r.status is not S.PASS:
            return None, r, f"{check_id} is {r.status.value}: {r.message}"
        m = r.details.get("measured")
        if isinstance(m, bool) or not isinstance(m, (int, float)) or not math.isfinite(m):
            return None, r, f"{check_id}: records no finite measured value"
        rec_unit = r.details.get("unit")
        if unit is not None and rec_unit is not None and rec_unit != unit:
            return None, r, f"{check_id}: measured in {rec_unit!r}, not {unit}"
        return float(m), r, None


def _evidence(results: Iterable[ValidationResult | None]) -> list[Evidence]:
    out: list[Evidence] = []
    seen: set[tuple[str | None, str | None]] = set()
    for r in results:
        for e in r.evidence if r is not None else []:
            key = (e.path, e.content_hash)
            if key not in seen:
                seen.add(key)
                out.append(e)
    return out


# --------------------------------------------------------------------------- rf.freq_plan


def _pointed(cx: _Ctx, target: str, lab_ids: set[str]) -> tuple[S, str]:
    """The status a response / gated row takes from one ``points_to`` entry, and why."""
    if target.startswith(LAB_PREFIX + "."):
        lab = target[len(LAB_PREFIX) + 1:]
        if lab in lab_ids:
            return S.NOT_VERIFIED, f"{target}: {NO_LAB_EVIDENCE}"
        return S.NOT_VERIFIED, f"{target}: names no lab item of the design"
    r, why = cx.result(target)
    if r is None:
        return S.NOT_VERIFIED, str(why)
    if r.status is S.NOT_APPLICABLE:
        return S.NOT_VERIFIED, f"{target} is NOT_APPLICABLE: it judges nothing here"
    return r.status, f"{target} {r.status.value}"


def plan_row(ir: CircuitIR, cx: _Ctx, line: Any, lab_ids: set[str]) -> dict:
    """One ``rf.freq_plan`` row (module docstring)."""
    row: dict[str, Any] = {"id": line.id, "kind": line.kind, "points_to": list(line.points_to), "note": line.note}
    f, why_f = value_in(line.f_hz, "Hz", f"{line.id}.f_hz")
    ref, why_ref = value_in(line.ref_hz, "Hz", f"{line.id}.ref_hz") if line.ref_hz is not None else (None, None)
    row.update(f_hz=f, ref_hz=ref)
    problems = [w for w in (why_f, why_ref) if w]
    if f is not None and ref is not None:
        row["offset_hz"] = f - ref
    if line.kind in ("response", "gated"):
        judged = [(t, *_pointed(cx, t, lab_ids)) for t in line.points_to]
        row["pointed"] = {t: s.value for t, s, _ in judged}
        status = worst_status(s for _, s, _ in judged) if judged else S.NOT_VERIFIED
        where = f"{_hz(f)}" if f is not None else line.id
        if ref is not None and f is not None:
            where += f" ({'+' if f >= ref else '-'}{_hz(abs(f - ref))} from {_hz(ref)})"
        row.update(status=status.value, reason=f"{line.kind} at {where}: " + "; ".join(w for _, _, w in judged))
        return row
    if problems:
        row.update(status=S.NOT_VERIFIED.value, reason="; ".join(problems))
        return row
    assert f is not None and ref is not None
    named = [(f"{line.id}.f_hz", line.f_hz), (f"{line.id}.ref_hz", line.ref_hz)]
    if line.kind == "margin":
        m, why_m = value_in(line.min_margin_hz, "Hz", f"{line.id}.min_margin_hz")
        if m is None:
            row.update(status=S.NOT_VERIFIED.value, reason=str(why_m))
            return row
        named.append((f"{line.id}.min_margin_hz", line.min_margin_hz))
        distance = abs(f - ref)
        ok = distance >= m - REL_TOL * max(abs(f), abs(ref))
        row.update(distance_hz=distance, min_margin_hz=m)
        text = f"|{_hz(f)} - {_hz(ref)}| = {_hz(distance)} {'>=' if ok else '<'} the margin {_hz(m)}"
    else:  # coincidence
        ok = not _same(f, ref)
        text = f"{_hz(f)} and {_hz(ref)} " + (f"differ by {_hz(abs(f - ref))}: no coincidence" if ok else "coincide")
    b = _bases(ir, named)
    if not b.confirmed:
        row.update(status=S.NOT_VERIFIED.value, reason=f"{text}, but it rests on unconfirmed {', '.join(b.weak)}")
    elif ok:
        row.update(status=S.PASS.value, reason=text)
    else:
        row.update(status=S.FAIL.value, reason=text + (f" ({line.note})" if line.note else ""), repair="human")
    return row


def freq_plan_result(ir: CircuitIR, cx: _Ctx) -> ValidationResult | None:
    """``rf.freq_plan`` (module docstring); ``None`` without a frequency plan."""
    rf = _rf(ir)
    lines = list(rf.frequency_plan) if rf is not None else []
    if not lines:
        return None
    lab_ids = {x.id for x in rf.lab_items}
    rows = [plan_row(ir, cx, line, lab_ids) for line in lines]
    status = _worst(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    head = f"{len(rows)} plan row(s) ({', '.join(f'{n} {s}' for s, n in sorted(counts.items()))}); arithmetic on the confirmed frequencies, not a measurement"
    rest = [f"{r['id']}: {r['reason']}" for r in rows if r["status"] != S.PASS.value]
    details: dict[str, Any] = {"rows": rows}
    if status is S.FAIL:
        details["repair"] = "human"
    return _result(FREQ_PLAN_CHECK, status, "; ".join([head, *rest]), **details)


# --------------------------------------------------------------------------- rf.regulatory_profile


def _profile_source(t: Traced | None) -> str:
    note = t.provenance.note if t is not None else None
    return note or "no source recorded"


def _limit_row(ir: CircuitIR, key: str, t: Traced, req_key: str, unit: str, found: dict, unusable: dict[str, str]) -> dict:
    row: dict[str, Any] = {"key": key, "requirement": req_key, "source": _profile_source(t)}
    limit, why = value_in(t, unit, key)
    if limit is None:
        row.update(status=S.NOT_VERIFIED.value, reason=str(why))
        return row
    row.update(limit=limit, unit=unit)
    if req_key in unusable:
        row.update(status=S.NOT_VERIFIED.value, reason=f"{req_key} is not usable: {unusable[req_key]}")
        return row
    if req_key not in found:
        row.update(status=S.NOT_APPLICABLE.value, reason=f"no confirmed {req_key} requirement to compare with {key}")
        return row
    value = float(found[req_key].traced.value)
    row["value"] = value
    text = f"{req_key} {value:.10g} {unit} vs the profile value {key} = {limit:.10g} {unit}"
    if value > limit * (1.0 + REL_TOL):
        b = basis(ir, key, t)
        if b.confirmed:
            row.update(status=S.FAIL.value, reason=f"{text}: exceeds it - two confirmed choices contradict each other (not a legal verdict)", repair="human")
        else:
            row.update(status=S.NOT_VERIFIED.value, reason=f"{text}: would exceed it, but the profile value is unconfirmed ({', '.join(b.weak)})")
    else:
        row.update(status=S.NOT_VERIFIED.value, reason=f"{text}: within an UNVERIFIED placeholder (never a PASS)")
    return row


def raster_channel(f: float, low: float, raster: float) -> tuple[int, bool]:
    """``(nearest channel index from 0, on the raster)`` of ``f`` on a raster of ``raster`` Hz counted from the first channel ``low``."""
    k = (f - low) / raster
    n = round(k)
    return int(n), abs(k - n) <= RASTER_TOL


def _band_rows(ir: CircuitIR, prefix: str, parts: dict[str, str], found: dict, unusable: dict[str, str]) -> list[dict]:
    keys = {s: parts[s] for s in (BAND_LOW, BAND_HIGH, RASTER) if s in parts}
    base: dict[str, Any] = {"key": ", ".join(keys.values()), "requirement": CARRIER_KEY,
                            "source": "; ".join(sorted({_profile_source(ir.parameters.get(k)) for k in keys.values()}))}
    values: dict[str, float] = {}
    problems: list[str] = []
    for s, k in keys.items():
        v, why = value_in(ir.parameters.get(k), "Hz", k)
        if v is None:
            problems.append(str(why))
        else:
            values[s] = v
    if problems:
        return [{**base, "status": S.NOT_VERIFIED.value, "reason": "; ".join(problems)}]
    if CARRIER_KEY in unusable:
        return [{**base, "status": S.NOT_VERIFIED.value, "reason": f"{CARRIER_KEY} is not usable: {unusable[CARRIER_KEY]}"}]
    if CARRIER_KEY not in found:
        return [{**base, "status": S.NOT_APPLICABLE.value, "reason": f"no confirmed {CARRIER_KEY} requirement to compare with the profile band"}]
    f = float(found[CARRIER_KEY].traced.value)
    b = _bases(ir, [(k, ir.parameters.get(k)) for k in keys.values()])
    unconfirmed = f" - but the profile value rests on unconfirmed {', '.join(b.weak)}" if not b.confirmed else ""
    rows: list[dict] = []
    low, high, raster = values.get(BAND_LOW), values.get(BAND_HIGH), values.get(RASTER)
    base["carrier_hz"] = f
    if low is not None and high is not None and low > high:
        return [{**base, "status": S.NOT_VERIFIED.value, "reason": f"the profile band is empty ({keys[BAND_LOW]} {_hz(low)} is above {keys[BAND_HIGH]} {_hz(high)})"}]
    outside = (low is not None and f < low * (1.0 - REL_TOL)) or (high is not None and f > high * (1.0 + REL_TOL))
    band_text = f"{_hz(low) if low is not None else '...'} - {_hz(high) if high is not None else '...'}"
    if low is not None or high is not None:
        if outside:
            fail = b.confirmed
            rows.append({**base, "check": "band", "status": (S.FAIL if fail else S.NOT_VERIFIED).value, "band_hz": [low, high],
                         "reason": f"{CARRIER_KEY} {_hz(f)} is outside the profile band {band_text} (a band is not a carrier frequency: state the channel){unconfirmed}",
                         **({"repair": "human"} if fail else {})})
        else:
            rows.append({**base, "check": "band", "status": S.NOT_VERIFIED.value, "band_hz": [low, high],
                         "reason": f"{CARRIER_KEY} {_hz(f)} is inside the UNVERIFIED profile band {band_text} (never a PASS)"})
    if raster is not None:
        if raster <= 0:
            rows.append({**base, "check": "raster", "status": S.NOT_VERIFIED.value, "reason": f"{keys[RASTER]} {raster!r} Hz is not a raster"})
        elif low is None:
            rows.append({**base, "check": "raster", "status": S.NOT_VERIFIED.value,
                         "reason": f"{keys[RASTER]} without {prefix}{BAND_LOW}: the raster's first channel is unknown"})
        elif not outside:
            n, on = raster_channel(f, low, raster)
            count = None
            if high is not None:
                m, whole = raster_channel(high, low, raster)
                count = m + 1 if whole else None
            if on:
                of = f" of {count}" if count is not None else ""
                rows.append({**base, "check": "raster", "status": S.NOT_VERIFIED.value, "channel": n + 1, "channels": count,
                             "reason": f"{CARRIER_KEY} {_hz(f)} is channel {n + 1}{of} of the UNVERIFIED {_hz(raster)} raster from {_hz(low)} (never a PASS)"})
            else:
                below, above = low + math.floor((f - low) / raster) * raster, low + math.ceil((f - low) / raster) * raster
                fail = b.confirmed
                rows.append({**base, "check": "raster", "status": (S.FAIL if fail else S.NOT_VERIFIED).value, "nearest_hz": [below, above],
                             "reason": f"{CARRIER_KEY} {_hz(f)} is not on the {_hz(raster)} raster from {_hz(low)} "
                                       f"(the nearest channels are {_hz(below)} and {_hz(above)}): state the channel{unconfirmed}",
                             **({"repair": "human"} if fail else {})})
    return rows


def profile_result(ir: CircuitIR) -> ValidationResult | None:
    """``rf.regulatory_profile`` (module docstring); ``None`` without profile keys. Never PASS."""
    from ai_eda.design.inputs import read_inputs  # the design package imports tools the validators also import

    rf = _rf(ir)
    keys = list(rf.profile_keys) if rf is not None else []
    if not keys:
        return None
    found, unusable = read_inputs(ir)
    rows: list[dict] = []
    bands: dict[str, dict[str, str]] = {}
    has_power_limit = False
    for key in keys:
        t = ir.parameters.get(key)
        suffix = next((s for s in (*PROFILE_LIMITS, BAND_LOW, BAND_HIGH, RASTER) if key.endswith(s) and len(key) > len(s)), None)
        if t is None:
            rows.append({"key": key, "status": S.NOT_VERIFIED.value, "reason": f"{key} is listed in ir.rf.profile_keys but has no value in ir.parameters"})
        elif suffix in PROFILE_LIMITS:
            req_key, unit = PROFILE_LIMITS[suffix]
            has_power_limit = has_power_limit or suffix == ".max_power"
            rows.append(_limit_row(ir, key, t, req_key, unit, found, unusable))
        elif suffix is not None:
            bands.setdefault(key[: -len(suffix)], {})[suffix] = key
        else:
            shown = t.value if not isinstance(t.value, float) else f"{t.value:.10g}"
            rows.append({"key": key, "status": S.NOT_VERIFIED.value, "value": t.value, "source": _profile_source(t),
                         "reason": f"{key} = {shown}{' ' + t.unit if t.unit else ''}: not compared (no requirement rule for it; an UNVERIFIED choice)"})
    for prefix, parts in bands.items():
        rows.extend(_band_rows(ir, prefix, parts, found, unusable))
    if has_power_limit:
        for key in RADIATED_KEYS:
            if key in found:
                rows.append({"key": key, "requirement": key, "status": S.NOT_VERIFIED.value, "value": float(found[key].traced.value),
                             "reason": f"{key} {float(found[key].traced.value):.10g} W is not compared with the profile's power limit: its reference point "
                                       "(the antenna port, conducted) is an assumption of the unverified profile; an ERP / EIRP number meets a conducted one "
                                       "only through calc.rf.erp_to_eirp / eirp_to_erp with a stated antenna gain"})
    status = S.FAIL if any(r["status"] == S.FAIL.value for r in rows) else S.NOT_VERIFIED
    exceed = [f"{r['key']}: {r['reason']}" for r in rows if r["status"] == S.FAIL.value]
    within = sum(1 for r in rows if r["status"] == S.NOT_VERIFIED.value)
    msg = f"{len(rows)} profile row(s), {len(exceed)} exceeded, {within} not a verdict; {UNVERIFIED_PROFILE}"
    if exceed:
        msg += "; " + "; ".join(exceed)
    details: dict[str, Any] = {"rows": rows, "profile_keys": keys, "unverified": True}
    if status is S.FAIL:
        details["repair"] = "human"
    return _result(PROFILE_CHECK, status, msg, **details)


# --------------------------------------------------------------------------- rf.model_grounding


def model_grounding_result(ir: CircuitIR) -> ValidationResult | None:
    """``rf.model_grounding`` (module docstring); ``None`` when the design has no model value."""
    rf = _rf(ir)
    listed = list(rf.model_values) if rf is not None else []
    unlisted = sorted(k for k in ir.parameters if k.startswith(MODEL_PREFIX) and k not in listed)
    if not listed and not unlisted:
        return None
    rows: list[dict] = []
    for key in [*listed, *unlisted]:
        t = ir.parameters.get(key)
        row: dict[str, Any] = {"key": key, "listed": key in listed}
        if t is None:
            row.update(status=S.NOT_VERIFIED.value, reason=f"{key} is listed in ir.rf.model_values but has no value in ir.parameters")
        else:
            p = t.provenance
            row.update(value=t.value, unit=t.unit, provenance=p.kind.value, note=p.note)
            if p.kind is ProvenanceKind.AUTHORITATIVE and p.source is not None:
                row.update(status=S.PASS.value, reason=f"grounded ({p.source.title})")
            else:
                why = {ProvenanceKind.USER_REQUIREMENT: "a confirmed model choice, not grounded",
                       ProvenanceKind.AUTHORITATIVE: "authoritative without a source document, not grounded"}.get(p.kind, f"{p.kind.value}, not grounded")
                row.update(status=S.NOT_VERIFIED.value, reason=why + ("" if key in listed else "; not listed in ir.rf.model_values"))
        rows.append(row)
    status = S.PASS if all(r["status"] == S.PASS.value for r in rows) else S.NOT_VERIFIED
    if status is S.PASS:
        msg = f"all {len(rows)} model value(s) grounded"
    else:
        open_ = [r["key"] for r in rows if r["status"] != S.PASS.value]
        msg = (f"{len(open_)} of {len(rows)} model value(s) not grounded ({', '.join(open_)}): every fixture / principle PASS resting on them is a verdict "
               f"{UNDER_MODEL_VALUES}; ground them with a datasheet fact or a lab record")
    return _result(MODEL_CHECK, status, msg, rows=rows)


# --------------------------------------------------------------------------- rf.deviation


@dataclass
class _Factor:
    name: str
    value: float | None = None
    unit: str = ""
    source: str = ""
    problem: str | None = None
    results: list[ValidationResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"factor": self.name, "value": self.value, "unit": self.unit, "source": self.source,
                "status": S.PASS.value if self.problem is None else S.NOT_VERIFIED.value, **({"reason": self.problem} if self.problem else {})}


def _same_point(a: Any, b: Any) -> bool:
    """Two phase expectations read the same quantity: the same drive, the same port read and the same frequency."""
    return a.drive == b.drive and a.to == b.to and _same(float(a.at.value), float(b.at.value))


def _phase_pairs(net: Any) -> tuple[list[tuple[Any, Any]], list[Any]]:
    """The ``(bias_lo, bias_hi)`` ``phase21_deg`` expectation pairs that read the same drive -> to at one frequency, and the ``bias_nom`` ones.

    A phase difference between two different nodes (or drives) is no chord
    slope of one tank, so a lo / hi pair is formed only from expectations of
    the same (drive, to, at).
    """
    phase = [e for e in net.expectations if e.quantity == "phase21_deg"]
    lo = [e for e in phase if e.state == PM_LO]
    hi = [e for e in phase if e.state == PM_HI]
    nom = [e for e in phase if e.state == PM_NOM]
    return [(a, b) for a in lo for b in hi if _same_point(a, b)], nom


def _fixture_check(net: Any, exp: Any) -> str:
    return f"{FIXTURE_PREFIX}.{net.id}" + (f".{exp.state}" if exp.state is not None else "") + f".{exp.id}"


def tank_slope(cx: _Ctx, net: Any) -> tuple[_Factor, dict[str, Any]]:
    """One phase-modulator tank's chord slope K_pm (rad/V) from its recorded ``bias_lo`` / ``bias_hi`` phases (module docstring)."""
    fac = _Factor(name=f"K_pm[{net.id}]", unit="rad/V")
    info: dict[str, Any] = {"network": net.id}
    states = {s.id: s for s in net.states}
    missing = [s for s in (PM_LO, PM_HI) if s not in states]
    if missing:
        fac.problem = f"network {net.id} has no state {', '.join(missing)}"
        return fac, info
    pairs, nom = _phase_pairs(net)
    if len(pairs) != 1:
        found = {st: [f"{e.id} ({e.drive}->{e.to} at {_hz(float(e.at.value))})" for e in net.expectations if e.quantity == "phase21_deg" and e.state == st]
                 for st in (PM_LO, PM_HI)}
        fac.problem = (f"network {net.id}: {'no' if not pairs else len(pairs)} {PM_LO} / {PM_HI} phase21_deg expectation pair(s) reading the same "
                       f"drive -> to at one frequency (exactly one is needed for the chord slope; {PM_LO}: {found[PM_LO] or 'none'}, "
                       f"{PM_HI}: {found[PM_HI] or 'none'})")
        return fac, info
    e_lo, e_hi = pairs[0]
    lo, hi = states[PM_LO].port_dc_v, states[PM_HI].port_dc_v
    ports = sorted(set(lo) | set(hi))
    one_sided = [p for p in ports if (p in lo) != (p in hi)]
    moving = [p for p in ports if p in lo and p in hi and not _same(float(lo[p].value), float(hi[p].value))]
    if one_sided or len(moving) != 1:
        fac.problem = (f"network {net.id}: the bias port is not one port whose DC level differs between {PM_LO} and {PM_HI} "
                       f"(differing: {moving or 'none'}; set in one state only: {one_sided or 'none'})")
        return fac, info
    port = moving[0]
    v_lo, v_hi = float(lo[port].value), float(hi[port].value)
    phis: dict[str, float] = {}
    for label, exp in ((PM_LO, e_lo), (PM_HI, e_hi)):
        check = _fixture_check(net, exp)
        m, r, why = cx.passed_number(check, "deg")
        if r is not None:
            fac.results.append(r)
        if m is None:
            fac.problem = str(why)
            return fac, info
        if r is not None and r.details.get("state") not in (None, label):
            fac.problem = f"{check} records state {r.details.get('state')!r}, not {label}"
            return fac, info
        recorded = {k: r.details.get(k) for k in ("drive", "to")} if r is not None else {}
        if any(v is not None and v != getattr(exp, k) for k, v in recorded.items()):
            fac.problem = f"{check} records {recorded.get('drive')}->{recorded.get('to')}, not the expectation's {exp.drive}->{exp.to}"
            return fac, info
        phis[label] = m
    slope = math.radians(phis[PM_HI] - phis[PM_LO]) / (v_hi - v_lo)
    fac.value = slope
    fac.source = f"({phis[PM_HI]:.6g} - {phis[PM_LO]:.6g}) deg / ({v_hi:.6g} - {v_lo:.6g}) V on port {port} at {_hz(float(e_lo.at.value))}"
    info.update(port=port, f_hz=float(e_lo.at.value), v_lo=v_lo, v_hi=v_hi, phi_lo_deg=phis[PM_LO], phi_hi_deg=phis[PM_HI], slope_rad_per_v=slope)
    e_nom = next((e for e in nom if _same_point(e, e_lo)), None)
    if e_nom is not None:
        r, _ = cx.result(_fixture_check(net, e_nom), numbers=True)
        m = r.details.get("measured") if r is not None else None
        if isinstance(m, (int, float)) and not isinstance(m, bool) and math.isfinite(m) and phis[PM_HI] != phis[PM_LO]:
            info["phi_nom_deg"] = float(m)
            info["linearity"] = (phis[PM_HI] + phis[PM_LO] - 2.0 * float(m)) / (phis[PM_HI] - phis[PM_LO])
            info["linearity_note"] = "(phi_hi + phi_lo - 2 phi_nom) / (phi_hi - phi_lo): recorded, no verdict (no distortion limit is a confirmed requirement)"
    return fac, info


def _param_factor(ir: CircuitIR, name: str, key: str, unit: str | None, unit_text: str) -> _Factor:
    t = ir.parameters.get(key)
    fac = _Factor(name=name, unit=unit_text, source=key)
    v, why = value_in(t, unit, key)
    if v is None:
        fac.problem = str(why)
    elif not v > 0:
        fac.problem = f"{key} = {v!r} is not > 0"
    else:
        b = basis(ir, key, t)
        fac.value = v
        if not b.confirmed:
            fac.problem = f"{key} rests on unconfirmed {', '.join(b.weak)}"
    return fac


def deviation_result(ir: CircuitIR, cx: _Ctx) -> ValidationResult | None:
    """``rf.deviation`` (module docstring); ``None`` on a design without a phase modulator or its audio chain."""
    from ai_eda.design.inputs import read_inputs  # the design package imports tools the validators also import

    rf = _rf(ir)
    pm_nets = [n for n in rf.networks if n.id.startswith(PM_NETWORK_PREFIX)] if rf is not None else []
    if not pm_nets and MODEL_K_PM_KEY not in ir.parameters and TAU_I_KEY not in ir.parameters:
        return None
    factors: list[_Factor] = [_param_factor(ir, "N", N_MULT_KEY, None, "")]
    tanks: list[dict[str, Any]] = []
    if pm_nets:
        slopes: list[_Factor] = []
        for net in pm_nets:
            fac, info = tank_slope(cx, net)
            slopes.append(fac)
            tanks.append(info)
        k = _Factor(name="K_pm", unit="rad/V", results=[r for f in slopes for r in f.results])
        bad = [f.problem for f in slopes if f.problem]
        if bad:
            k.problem = "; ".join(str(b) for b in bad)
        else:
            total = abs(sum(float(f.value) for f in slopes))  # type: ignore[arg-type]
            k.source = " + ".join(f"{f.name} {float(f.value):.6g}" for f in slopes)  # type: ignore[arg-type]
            if total > 0:
                k.value = total
            else:
                k.problem = f"the tanks' phase does not move with the bias ({k.source} rad/V): the chain gives no deviation"
        factors.append(k)
    else:
        k = _param_factor(ir, "K_pm", MODEL_K_PM_KEY, "rad/V", "rad/V")
        if k.problem is None:
            k.problem = (f"K_pm is the model constant {MODEL_K_PM_KEY}, not a verified network slope "
                         f"(no {PM_NETWORK_PREFIX}* fixture network on this board)")
        factors.append(k)
    a = _Factor(name="a", unit="(ratio)")
    couples: list[tuple[str, float]] = []
    problems: list[str] = []
    for check in PM_COUPLE_CHECKS:
        m, r, why = cx.passed_number(check, "dB")
        if r is not None:
            a.results.append(r)
        if m is None:
            problems.append(str(why))
        else:
            couples.append((check, m))
    if problems:
        a.problem = "; ".join(problems)
    else:
        worst_check, db = max(couples, key=lambda x: x[1])
        a.value = 10.0 ** (db / 20.0)
        a.source = f"the largest of {', '.join(f'{c} {m:.4g} dB' for c, m in couples)} ({worst_check}: the over-deviation side)"
    factors.append(a)
    v = _Factor(name="V_max", unit="V", source=PM_DRIVE_CHECK)
    m, r, why = cx.passed_number(PM_DRIVE_CHECK, "V")
    if r is not None:
        v.results.append(r)
    if m is None:
        v.problem = why
    elif not m > 0:
        v.problem = f"{PM_DRIVE_CHECK} measured {m!r} V, not > 0"
    else:
        v.value = m
    factors.append(v)
    tau = _param_factor(ir, "tau_i", TAU_I_KEY, "s", "s")
    checked, r, why = cx.passed_number(INTEGRATOR_CHECK, None)
    if r is not None:
        tau.results.append(r)
    if checked is None:
        unchecked = f"tau_i is not checked by the integrator's AC: {why}"
        tau.problem = unchecked if tau.problem is None else f"{tau.problem}; {unchecked}"
    else:
        tau.source = f"{TAU_I_KEY}, checked by {INTEGRATOR_CHECK}"
    factors.append(tau)
    details: dict[str, Any] = {"kind": DEVIATION_KIND, "formula": "Delta_f = N * K_pm * a * V_max / (2 pi tau_i)", "factors": [f.as_dict() for f in factors],
                               "tanks": tanks}
    evidence = _evidence(r for f in factors for r in f.results)
    values = {f.name: f.value for f in factors}
    if all(v is not None for v in values.values()):
        delta = float(values["N"]) * float(values["K_pm"]) * float(values["a"]) * float(values["V_max"]) / (2.0 * math.pi * float(values["tau_i"]))  # type: ignore[arg-type]
        details["delta_f_hz"] = delta
    else:
        delta = None
    open_ = [f"{f.name}: {f.problem}" for f in factors if f.problem]
    if open_:
        if delta is not None:
            details["delta_f_note"] = "computed from the factors, not a verdict: " + "; ".join(open_)
        head = f"peak deviation {_hz(delta)} (not a verdict)" if delta is not None else "peak deviation not computed"
        return _result(DEVIATION_CHECK, S.NOT_VERIFIED, f"{head}: " + "; ".join(open_) + "; the real deviation is a lab item", evidence, **details)
    assert delta is not None
    found, unusable = read_inputs(ir)
    if DEVIATION_KEY in unusable:
        return _result(DEVIATION_CHECK, S.NOT_VERIFIED, f"peak deviation {_hz(delta)} {UNDER_MODEL_VALUES}, but {DEVIATION_KEY} is not usable: {unusable[DEVIATION_KEY]}", evidence, **details)
    if DEVIATION_KEY not in found:
        return _result(DEVIATION_CHECK, S.NOT_VERIFIED, f"peak deviation {_hz(delta)} {UNDER_MODEL_VALUES}, but there is no confirmed {DEVIATION_KEY} requirement to compare", evidence, **details)
    limit = float(found[DEVIATION_KEY].traced.value)
    details.update(requirement_hz=limit, fraction_of_requirement=delta / limit if limit > 0 else None)
    text = f"peak deviation {_hz(delta)} = N {values['N']:.6g} x K_pm {values['K_pm']:.6g} rad/V x a {values['a']:.6g} x V_max {values['V_max']:.6g} V / (2 pi tau_i {values['tau_i']:.6g} s)"
    if delta <= limit * (1.0 + REL_TOL):
        return _result(DEVIATION_CHECK, S.PASS, f"{text} <= {DEVIATION_KEY} {_hz(limit)} {UNDER_MODEL_VALUES}; the real deviation is a lab item", evidence, **details)
    details["repair"] = "human"
    return _result(DEVIATION_CHECK, S.FAIL, f"{text} > {DEVIATION_KEY} {_hz(limit)} {UNDER_MODEL_VALUES}: the verified factors over-deviate", evidence, **details)


# --------------------------------------------------------------------------- rf.lab.<id>


def lab_results(ir: CircuitIR) -> list[ValidationResult]:
    """One NOT_VERIFIED ``rf.lab.<id>`` per lab item: nothing here can measure."""
    rf = _rf(ir)
    out: list[ValidationResult] = []
    for item in rf.lab_items if rf is not None else []:
        instruments = f" [{', '.join(item.instruments)}]" if item.instruments else ""
        out.append(_result(f"{LAB_PREFIX}.{item.id}", S.NOT_VERIFIED, f"{NO_LAB_EVIDENCE}: {item.what}{instruments} - {item.reason}",
                           what=item.what, block=item.block, instruments=list(item.instruments), reason=item.reason,
                           evidence_path="none: no lab-evidence importer exists in this version; a measurement is made and recorded by a human, never computed here"))
    return out


# --------------------------------------------------------------------------- block.interface.<net>


def _agree(ir: CircuitIR, members: list[tuple[Any, Any]], attr: str, unit: str, what: str) -> dict | None:
    """One agreement row over the ports' ``attr`` (``None`` when no port states it)."""
    stated = [(b.id, p.name, getattr(p, attr)) for b, p in members if getattr(p, attr) is not None]
    if not stated:
        return None
    silent = [f"{b.id}.{p.name}" for b, p in members if getattr(p, attr) is None]
    if silent:
        return {"check": what, "status": S.NOT_VERIFIED.value, "reason": f"{what} stated by {', '.join(f'{b}.{n}' for b, n, _ in stated)} but not by {', '.join(silent)}"}
    numbers: list[tuple[str, float]] = []
    for b, n, t in stated:
        v, why = value_in(t, unit, f"{b}.{n}.{attr}")
        if v is None:
            return {"check": what, "status": S.NOT_VERIFIED.value, "reason": str(why)}
        numbers.append((f"{b}.{n}", v))
    text = ", ".join(f"{label} {v:.10g} {unit}" for label, v in numbers)
    same = all(_same(v, numbers[0][1]) for _, v in numbers)
    b = _bases(ir, [(f"{bid}.{n}.{attr}", t) for bid, n, t in stated])
    row: dict[str, Any] = {"check": what, "values": {label: v for label, v in numbers}}
    if not b.confirmed:
        row.update(status=S.NOT_VERIFIED.value, reason=f"{what}: {text}, resting on unconfirmed {', '.join(b.weak)}")
    elif same:
        row.update(status=S.PASS.value, reason=f"{what} agrees: {text}")
    else:
        row.update(status=S.FAIL.value, reason=f"{what} disagrees: {text}", repair="human")
    return row


def interface_results(ir: CircuitIR) -> list[ValidationResult]:
    """``block.interface.<net>`` for every net that ports of two or more blocks share (module docstring)."""
    rf = _rf(ir)
    groups: dict[str, list[tuple[Any, Any]]] = {}
    for b in rf.blocks if rf is not None else []:
        for p in b.ports:
            groups.setdefault(p.net, []).append((b, p))
    out: list[ValidationResult] = []
    for net, members in groups.items():
        if len({b.id for b, _ in members}) < 2:
            continue
        names = [f"{b.id}.{p.name}" for b, p in members]
        rows: list[dict] = []
        if ir.net(net) is None:
            rows.append({"check": "net", "status": S.FAIL.value, "reason": f"net {net} is not a net of the design", "repair": "human"})
        kinds = sorted({p.kind for _, p in members})
        rows.append({"check": "kind", "status": (S.PASS if len(kinds) == 1 else S.FAIL).value,
                     "reason": f"kind {kinds[0]}" if len(kinds) == 1 else "kinds disagree: " + ", ".join(f"{b.id}.{p.name} {p.kind}" for b, p in members),
                     **({} if len(kinds) == 1 else {"repair": "human"})})
        refs = sorted({p.reference_net for _, p in members})
        rows.append({"check": "reference_net", "status": (S.PASS if len(refs) == 1 else S.FAIL).value,
                     "reason": f"reference {refs[0]}" if len(refs) == 1 else "reference nets disagree: " + ", ".join(f"{b.id}.{p.name} {p.reference_net}" for b, p in members),
                     **({} if len(refs) == 1 else {"repair": "human"})})
        for attr, unit, what in (("z0_ohm", "ohm", "z0"), ("frequency_hz", "Hz", "frequency"), ("voltage_v", "V", "voltage")):
            row = _agree(ir, members, attr, unit, what)
            if row is not None:
                rows.append(row)
        outs = [f"{b.id}.{p.name}" for b, p in members if p.direction == "out"]
        drivers = [f"{b.id}.{p.name}" for b, p in members if p.direction in ("out", "bidir")]
        if len(outs) > 1:
            rows.append({"check": "direction", "status": S.FAIL.value, "reason": f"{len(outs)} outputs drive {net}: {', '.join(outs)}", "repair": "human"})
        elif not drivers:
            rows.append({"check": "direction", "status": S.NOT_VERIFIED.value, "reason": f"no block port drives {net} (all of {', '.join(names)} are inputs)"})
        else:
            rows.append({"check": "direction", "status": S.PASS.value, "reason": f"driven by {', '.join(outs or drivers)}"})
        status = _worst(rows)
        bad = [r["reason"] for r in rows if r["status"] != S.PASS.value]
        msg = f"{net} between {', '.join(names)}: " + ("; ".join(bad) if bad else "; ".join(r["reason"] for r in rows)) + " (IR arithmetic)"
        details: dict[str, Any] = {"net": net, "ports": names, "rows": rows}
        if status is S.FAIL:
            details["repair"] = "human"
        out.append(_result(f"{INTERFACE_PREFIX}.{net}", status, msg, **details))
    return out


# --------------------------------------------------------------------------- power.rail_budget / power.headroom


def call_calculator(tool: str, inputs: dict[str, Traced], ids: dict[str, str]) -> Traced | str:
    """``tool`` called by role through :data:`~ai_eda.tools.calc.recompute.CALCULATORS`, or why it cannot be."""
    entry = CALCULATORS.get(tool)
    if entry is None:
        return f"{tool} is not a registered calculator in this build"
    fn, roles = entry
    unknown = [r for r in roles if r not in inputs]
    if unknown:
        return f"{tool} takes the roles {list(roles)}; {unknown} are not known here"
    try:
        return fn(*(inputs[r] for r in roles), tuple(ids[r] for r in roles))
    except (ValueError, ArithmeticError, TypeError) as e:  # ArithmeticError: a zero division or an OverflowError
        return f"{tool} refused the inputs: {e}"


def _zero(unit: str, why: str) -> Traced:
    """An unstated input taken at 0 - only ever to bound a headroom from above."""
    return assumption(0.0, why, unit)


def rail_budget_results(ir: CircuitIR) -> list[ValidationResult]:
    """``power.rail_budget.<rail>`` (module docstring)."""
    rf = _rf(ir)
    out: list[ValidationResult] = []
    for rail in rf.rails if rf is not None else []:
        check = f"{RAIL_BUDGET_PREFIX}.{rail.rail}"
        here = f"rf.rails[{rail.rail}]"
        details: dict[str, Any] = {"rail": rail.rail, "regulator": rail.regulator_ref, "i_min_a": float(rail.i_min.value), "i_max_a": float(rail.i_max.value)}
        if rail.i_rating is None:
            out.append(_result(check, S.NOT_VERIFIED, f"{rail.rail}: no current rating stated for {rail.regulator_ref} (the load is {float(rail.i_min.value):.4g}-{float(rail.i_max.value):.4g} A)", **details))
            continue
        details["i_rating_a"] = float(rail.i_rating.value)
        margin = call_calculator(RAIL_BUDGET_CALC, {"i_load": rail.i_max, "i_rating": rail.i_rating}, {"i_load": f"{here}.i_max", "i_rating": f"{here}.i_rating"})
        if isinstance(margin, str):
            out.append(_result(check, S.NOT_VERIFIED, f"{rail.rail}: {margin}", **details))
            continue
        m = float(margin.value)
        details.update(margin_a=m, calculator=margin.provenance.tool, calc_version=margin.provenance.tool_version, formula=margin.provenance.note)
        b = _bases(ir, [(f"{here}.i_max", rail.i_max), (f"{here}.i_rating", rail.i_rating)])
        text = f"{rail.rail} ({rail.regulator_ref}): rating {float(rail.i_rating.value):.4g} A - load up to {float(rail.i_max.value):.4g} A = {m:.4g} A"
        if m < 0:
            if b.confirmed:
                details["repair"] = "human"
                out.append(_result(check, S.FAIL, f"{text}: the load exceeds the rating - the confirmed values contradict each other", **details))
            else:
                out.append(_result(check, S.NOT_VERIFIED, f"{text}: would exceed the rating, but rests on unconfirmed {', '.join(b.weak)}", **details))
        elif b.grounded:
            out.append(_result(check, S.PASS, f"{text} (grounded currents)", **details))
        else:
            out.append(_result(check, S.NOT_VERIFIED, f"{text}: not a verdict - ungrounded {', '.join(b.weak + b.ungrounded)} (currents and ratings are datasheet facts)", **details))
    return out


def _feeding_rail(ir: CircuitIR, rail: Any, rails: list[Any]) -> tuple[Any | None, str | None]:
    """The other budget rail a regulator's pins touch (it is fed from it), ``(None, None)`` for none, ``(None, why)`` when unclear."""
    nets = {n.name for n in ir.nets for p in n.pins if p.component_ref == rail.regulator_ref}
    others = [r for r in rails if r.rail != rail.rail and r.rail in nets]
    if len(others) > 1:
        return None, f"{rail.regulator_ref}'s pins touch several budget rails ({', '.join(r.rail for r in others)}): which one feeds it is not stated"
    return (others[0], None) if others else (None, None)


def headroom_results(ir: CircuitIR) -> list[ValidationResult]:
    """``power.headroom.<regulator_ref>`` (module docstring)."""
    rf = _rf(ir)
    rails = list(rf.rails) if rf is not None else []
    out: list[ValidationResult] = []
    for rail in rails:
        check = f"{HEADROOM_PREFIX}.{rail.regulator_ref}"
        here = f"rf.rails[{rail.rail}]"
        details: dict[str, Any] = {"rail": rail.rail, "regulator": rail.regulator_ref}
        if ir.component(rail.regulator_ref) is None:
            out.append(_result(check, S.NOT_VERIFIED, f"{rail.regulator_ref} ({rail.rail}) is not a component of the design", **details))
            continue
        feed, why = _feeding_rail(ir, rail, rails)
        if why is not None:
            out.append(_result(check, S.NOT_VERIFIED, why, **details))
            continue
        if feed is not None:
            v_in, v_in_id, v_in_text = feed.v_out, f"rf.rails[{feed.rail}].v_out", f"the output of {feed.regulator_ref} ({feed.rail})"
        else:
            v_in, v_in_id, v_in_text = ir.parameters.get(PACK_CUTOFF_KEY), PACK_CUTOFF_KEY, f"the pack cut-off {PACK_CUTOFF_KEY}"
            if v_in is None:
                out.append(_result(check, S.NOT_VERIFIED, f"{rail.regulator_ref} ({rail.rail}): no {PACK_CUTOFF_KEY}: the minimum input voltage is not stated", **details))
                continue
        details["v_in_from"] = v_in_id
        unknown: list[str] = []
        dropout, r_path = rail.dropout_v, rail.path_r_ohm
        if dropout is None:
            unknown.append("the dropout")
            dropout = _zero("V", "dropout not stated: 0 V bounds the headroom from above")
        if r_path is None:
            unknown.append("the path resistance")
            r_path = _zero("ohm", "path resistance not stated: 0 ohm bounds the headroom from above")
        ids = {"v_in_min": v_in_id, "v_out": f"{here}.v_out", "v_dropout": f"{here}.dropout_v", "i_load": f"{here}.i_max", "r_path": f"{here}.path_r_ohm"}
        h = call_calculator(HEADROOM_CALC, {"v_in_min": v_in, "v_out": rail.v_out, "v_dropout": dropout, "i_load": rail.i_max, "r_path": r_path}, ids)
        if isinstance(h, str):
            out.append(_result(check, S.NOT_VERIFIED, f"{rail.regulator_ref} ({rail.rail}): {h}", **details))
            continue
        head = float(h.value)
        details.update(headroom_v=head, v_in_min_v=float(v_in.value), v_out_v=float(rail.v_out.value), dropout_v=float(dropout.value), i_load_a=float(rail.i_max.value),
                       r_path_ohm=float(r_path.value), unstated=unknown, calculator=h.provenance.tool, calc_version=h.provenance.tool_version, formula=h.provenance.note)
        stated = [(v_in_id, v_in), (f"{here}.v_out", rail.v_out), (f"{here}.i_max", rail.i_max)]
        facts = [(f"{here}.i_max", rail.i_max)]
        for name, t in ((f"{here}.dropout_v", rail.dropout_v), (f"{here}.path_r_ohm", rail.path_r_ohm)):
            if t is not None:
                stated.append((name, t))
                facts.append((name, t))
        b_stated, b_facts = _bases(ir, stated), _bases(ir, facts)
        text = (f"{rail.regulator_ref} ({rail.rail}) at {v_in_text} {float(v_in.value):.4g} V: {float(v_in.value):.4g} - {float(rail.i_max.value):.4g} A x "
                f"{float(r_path.value):.4g} ohm - {float(dropout.value):.4g} V - {float(rail.v_out.value):.4g} V = {head:+.4g} V")
        bound = f" (with {' and '.join(unknown)} at 0, an upper bound)" if unknown else ""
        if head < 0:
            if b_stated.confirmed:
                details["repair"] = "human"
                out.append(_result(check, S.FAIL, f"{text}{bound}: out of regulation at the minimum input - the confirmed values contradict each other", **details))
            else:
                out.append(_result(check, S.NOT_VERIFIED, f"{text}{bound}: would be out of regulation, but rests on unconfirmed {', '.join(b_stated.weak)}", **details))
        elif unknown:
            out.append(_result(check, S.NOT_VERIFIED, f"{text}{bound}: {' and '.join(unknown)} not stated - not a verdict", **details))
        elif b_stated.confirmed and b_facts.grounded:
            out.append(_result(check, S.PASS, f"{text} (grounded currents, dropout and path resistance)", **details))
        else:
            weak = b_stated.weak + [x for x in b_facts.weak + b_facts.ungrounded if x not in b_stated.weak]
            out.append(_result(check, S.NOT_VERIFIED, f"{text}: not a verdict - ungrounded {', '.join(weak)} (currents, dropout and path resistance are datasheet facts)", **details))
    return out


# --------------------------------------------------------------------------- retirement and the validator


def _owned(check_id: str) -> bool:
    return any(check_id.startswith(p) for p in OWNED_PREFIXES)


def retired_results(ir: CircuitIR, produced: set[str]) -> list[ValidationResult]:
    """A superseding NOT_APPLICABLE for every recorded check id of this validator the current IR no longer produces."""
    out: list[ValidationResult] = []
    for check_id, last in ir.validation.latest_by_check().items():
        if not _owned(check_id) or check_id in produced or last.status is S.NOT_APPLICABLE:
            continue
        out.append(_result(check_id, S.NOT_APPLICABLE, "no longer produced by the design's RF content (superseded)", superseded=last.status.value))
    return out


def rf_results(ir: CircuitIR) -> list[ValidationResult]:
    """Every RF check result for ``ir`` (module docstring), plus the retirement of the ones it no longer produces."""
    out: list[ValidationResult] = []
    if _rf(ir) is not None:
        cx = _Ctx(ir=ir, current=ir.content_hash())
        for r in (freq_plan_result(ir, cx), profile_result(ir), model_grounding_result(ir), deviation_result(ir, cx)):
            if r is not None:
                out.append(r)
        out += lab_results(ir)
        out += interface_results(ir)
        out += rail_budget_results(ir)
        out += headroom_results(ir)
    out += retired_results(ir, {r.check_id for r in out})
    return out


class RFChecksValidator(Validator):
    """The RF design checks (module docstring): on a design with ``ir.rf``, again after SPICE, and to retire its old results."""

    id = RF_TOOL
    description = ("RF design checks of ir.rf: frequency-plan arithmetic, the unverified regulatory profile (never PASS), model-value grounding, "
                   "the deviation chain from recorded SPICE verdicts, lab items, block interfaces and rail budgets")
    consumes = frozenset({SPICE_CHECK_ID})

    def applies_to(self, ir: CircuitIR) -> bool:
        if _rf(ir) is not None:
            return True
        return any(_owned(k) and r.status is not S.NOT_APPLICABLE for k, r in ir.validation.latest_by_check().items())

    def validate(self, ir: CircuitIR, ctx: ValidationContext) -> list[ValidationResult]:
        return rf_results(ir)


default_registry.register(RFChecksValidator())

__all__ = [
    "DEVIATION_CHECK",
    "FIXTURE_PREFIX",
    "FREQ_PLAN_CHECK",
    "HEADROOM_PREFIX",
    "INTERFACE_PREFIX",
    "LAB_PREFIX",
    "MODEL_CHECK",
    "NO_LAB_EVIDENCE",
    "OWNED_PREFIXES",
    "PROFILE_CHECK",
    "RAIL_BUDGET_PREFIX",
    "RF_TOOL",
    "RF_VERSION",
    "RFChecksValidator",
    "UNDER_MODEL_VALUES",
    "UNVERIFIED_PROFILE",
    "Basis",
    "basis",
    "call_calculator",
    "deviation_result",
    "freq_plan_result",
    "headroom_results",
    "interface_results",
    "lab_results",
    "model_grounding_result",
    "plan_row",
    "profile_result",
    "rail_budget_results",
    "raster_channel",
    "retired_results",
    "rf_results",
    "tank_slope",
    "value_in",
]
