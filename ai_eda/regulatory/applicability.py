"""Deterministic applicability: a declarative rule, the user's scope answers and the IR's requirements - no model.

Invariants this module enforces:

* **The decision is a pure function.** :func:`evaluate` reads only the rule,
  the answers and the requirements it is given and returns the same
  :class:`Evaluation` for the same inputs. Nothing is guessed: an input the
  rule needs and does not find is reported as a missing key
  (``USER_INPUT_REQUIRED`` naming it), never defaulted.
* **Only the user's and official facts decide.** A requirement is read only
  when its value's provenance is ``user_requirement`` or ``authoritative``;
  an ``llm_generated`` extraction the user has not confirmed, a model
  ``assumption`` or a ``derived`` figure is reported as a missing input
  naming the requirement and how to settle it (confirm it, or answer the
  key directly) - a scope decision never rests on a guess.
* **Three-valued logic.** A rule is ``APPLICABLE``, ``NOT_APPLICABLE`` or
  ``UNDECIDED``. ``all_of`` is ``NOT_APPLICABLE`` as soon as one member is
  (a 12 V DC design is outside the LVD whatever the radio answer),
  ``APPLICABLE`` only when every member is; ``any_of`` mirrors it; ``not``
  swaps the decided values and keeps ``UNDECIDED``.
* **A yes/no question needs a yes/no answer.** An ``answer`` rule whose
  expected values are all yes/no leaves the rule ``UNDECIDED`` (naming the
  key) when the given answer is neither - ``wifi and bluetooth`` is not
  ``no``; free-text expectations (``consumer`` / ``professional``) compare
  the normalised text.
* **Voltages are read by the quantity parser.** A ``voltage_range`` rule
  rates the *equipment*: it reads the named requirement (``input_voltage``),
  every extra requirement key the rule lists (``output_voltage``), an answer
  under any of those keys, and - when the rule names a ``scope_answer`` such
  as ``highest_rated_voltage`` - the user's statement of the highest voltage
  rating anywhere in the product. Each is a numeric ``Traced`` in volts, a
  ``[low, high]`` range, or a typed string such as ``"12 V DC"`` read with
  :func:`ai_eda.tools.calc.quantity.parse_quantity`. One voltage inside the
  band makes the rule ``APPLICABLE``; ``NOT_APPLICABLE`` needs every voltage
  outside the band *and*, when a ``scope_answer`` is named, that answer given
  - an input rail alone never proves a product is out of scope. AC or DC
  comes from the value's own words next to the volt unit for English
  (``12 V DC``, ``230VAC``, ``DC 60 V``, ``60 V (DC)``; the word ``AC`` in
  prose such as "from an AC adapter" is not a rating); the Korean words
  ``교류`` / ``직류`` count anywhere in the value string or the requirement's
  own sentence (never in a provenance note); else from the ``mains_powered`` answer (yes =
  AC, no = DC); neither -> missing ``mains_powered``. When the words say one
  kind and the mains answer the other, the contradiction is a missing input
  (``mains_powered``) - never resolved silently in either direction. A value
  in another unit (or one the parser can not read) is a missing input with
  the reason, not a number. The provenance *note* of a value is never read:
  it carries the model's own sentence.
* **The rationale states the inputs.** Every decision carries the values it
  read and where they came from (``12 V DC (requirement req.input_voltage)``),
  the band it compared against and, when the rule cites a grounding quote,
  the section label - so the reviewer and the user can retrace it.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from pydantic import BaseModel, Field

from ai_eda.ir.provenance import ProvenanceKind
from ai_eda.ir.regulatory import Applicability
from ai_eda.ir.requirements import Requirement
from ai_eda.ir.validation import ValidationStatus
from ai_eda.llm.extraction import CONFIRM_ANSWERS, NEGATIVE_ANSWERS, normalise_answer
from ai_eda.regulatory.candidates import ApplicabilityRule
from ai_eda.tools.calc.quantity import Quantity, QuantityRange, parse_quantity, parse_unit

APPLICABILITY_VERSION = "2"

#: the scope answer that decides AC vs DC when the requirement does not say
MAINS_KEY = "mains_powered"
#: provenance kinds a rule may read a requirement value from
TRUSTED_KINDS: frozenset[ProvenanceKind] = frozenset({ProvenanceKind.USER_REQUIREMENT, ProvenanceKind.AUTHORITATIVE})

#: ``AC`` / ``DC`` next to the value: after the volt unit (``12 V DC``, ``230VAC``, ``48 Vdc``, ``60 V (DC)``) or right before
#: the number (``DC 60 V``, ``AC220V``), or the Korean words; never the bare word in prose (``AC adapter``)
_AC_RE = re.compile(r"(?<![A-Za-z])[Vv]\s*\(?\s*(?:AC|ac|Ac)\s*\)?(?![A-Za-z])|(?<![A-Za-z])(?:AC|ac|Ac)\s*(?=[0-9])|교류")
_DC_RE = re.compile(r"(?<![A-Za-z])[Vv]\s*\(?\s*(?:DC|dc|Dc)\s*\)?(?![A-Za-z])|(?<![A-Za-z])(?:DC|dc|Dc)\s*(?=[0-9])|직류")
_YES_EXTRA = frozenset({"true", "1", "있음", "있어요", "포함"})
_NO_EXTRA = frozenset({"false", "0", "없음", "없어요", "미포함", "아님"})


def yes_no(answer: str | None) -> str | None:
    """``"yes"`` / ``"no"`` for a yes-like / no-like answer (``y``, ``네``, ``아니오`` ...), else ``None``."""
    norm = normalise_answer(answer)
    if not norm:
        return None
    if norm in CONFIRM_ANSWERS or norm in _YES_EXTRA:
        return "yes"
    if norm in NEGATIVE_ANSWERS or norm in _NO_EXTRA:
        return "no"
    return None


class MissingInput(BaseModel):
    key: str
    reason: str


class Evaluation(BaseModel):
    """What :func:`evaluate` decided and why."""

    applicability: Applicability
    #: the decision as a validation status: NOT_APPLICABLE / NOT_VERIFIED (applies; compliance not verified) /
    #: USER_INPUT_REQUIRED (an input is missing) - never PASS
    status: ValidationStatus
    rationale: str
    #: input key -> "value (where it came from)"
    inputs_used: dict[str, str] = Field(default_factory=dict)
    missing: list[MissingInput] = Field(default_factory=list)
    #: grounding-quote section labels the decision rests on (from the rules that decided it)
    evidence: list[str] = Field(default_factory=list)

    @property
    def missing_keys(self) -> list[str]:
        out: list[str] = []
        for m in self.missing:
            if m.key not in out:
                out.append(m.key)
        return out


def status_for(applicability: Applicability) -> ValidationStatus:
    if applicability is Applicability.NOT_APPLICABLE:
        return ValidationStatus.NOT_APPLICABLE
    if applicability is Applicability.APPLICABLE:
        return ValidationStatus.NOT_VERIFIED
    return ValidationStatus.USER_INPUT_REQUIRED


# --------------------------------------------------------------------------- inputs


def _requirement(requirements: Iterable[Requirement] | Any, key: str) -> Requirement | None:
    if requirements is None:
        return None
    getter = getattr(requirements, "get", None)
    if callable(getter) and not isinstance(requirements, dict):
        found = getter(key)
        return found if isinstance(found, Requirement) else None
    if isinstance(requirements, dict):
        found = requirements.get(key)
        return found if isinstance(found, Requirement) else None
    for r in requirements:
        if getattr(r, "key", None) == key:
            return r
    return None


def _current_kind(*texts: str | None) -> str | None:
    """``"ac"`` / ``"dc"`` when exactly one of them is stated next to the unit in the texts, else ``None``."""
    joined = " ".join(t for t in texts if t)
    ac, dc = bool(_AC_RE.search(joined)), bool(_DC_RE.search(joined))
    if ac and not dc:
        return "ac"
    if dc and not ac:
        return "dc"
    return None


def _to_volts(value: Any, unit: str | None) -> tuple[float | list[float] | None, str | None]:
    """A Traced payload in volts, or ``(None, reason)``."""
    if unit is None:
        return None, "no unit"
    parsed = parse_unit(unit)
    if parsed is None or parsed[0] != "V":
        return None, f"unit {unit!r} is not a voltage"
    scale = 10.0 ** parsed[1]
    if isinstance(value, bool):
        return None, "boolean value"
    if isinstance(value, (int, float)):
        return float(value) * scale, None
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        lo, hi = sorted(float(v) * scale for v in value)
        return [lo, hi], None
    return None, f"value {value!r} is not a number or a [low, high] range"


def _parse_text(text: str) -> tuple[float | list[float] | None, str | None]:
    q = parse_quantity(text)
    if q is None:
        return None, f"{text!r} is not one unambiguous quantity with a unit"
    if q.unit != "V":
        return None, f"{text!r} is a {q.unit} quantity, not a voltage"
    if isinstance(q, QuantityRange):
        return [q.low, q.high], None
    assert isinstance(q, Quantity)
    return q.value, None


def _untrusted_reason(req: Requirement) -> str:
    kind = req.value.provenance.kind if req.value is not None else None
    what = {
        ProvenanceKind.LLM_GENERATED: "a model extraction / inference the user has not confirmed",
        ProvenanceKind.ASSUMPTION: "a model assumption",
        ProvenanceKind.DERIVED: "a derived figure",
    }.get(kind, str(kind))  # type: ignore[arg-type]
    return (f"requirement {req.id} is {what} ({kind}), and a scope decision may not rest on it: confirm it "
            f"(confirm_requirements / accept_implicit) or answer {req.key} directly")


class _Voltage(BaseModel):
    key: str
    volts: float | list[float]
    kind: str | None = None
    origin: str


def _read_voltage(key: str, answers: dict[str, str], requirements: Any, *, required: bool) -> tuple[_Voltage | None, MissingInput | None]:
    """The voltage under ``key``: a trusted requirement, else an answer, else (when ``required`` or the requirement is unusable) a missing input."""
    req = _requirement(requirements, key)
    if req is not None and req.value is not None and req.value.provenance.kind in TRUSTED_KINDS:
        raw = req.value.value
        volts, why = _parse_text(raw) if isinstance(raw, str) else _to_volts(raw, req.value.unit)
        if volts is None:
            return None, MissingInput(key=key, reason=f"requirement {req.id} can not be read as a voltage: {why}")
        return _Voltage(key=key, volts=volts, kind=_current_kind(raw if isinstance(raw, str) else None, req.text), origin=f"requirement {req.id}"), None
    if key in answers and answers[key].strip():
        volts, why = _parse_text(answers[key])
        if volts is None:
            return None, MissingInput(key=key, reason=f"answer {key}={answers[key]!r} can not be read as a voltage: {why}")
        return _Voltage(key=key, volts=volts, kind=_current_kind(answers[key]), origin=f"answer {key}"), None
    if req is not None and req.value is not None:
        return None, MissingInput(key=key, reason=_untrusted_reason(req))
    if required:
        return None, MissingInput(key=key, reason=f"no requirement or answer {key!r} states the design's rated voltage")
    return None, None


def _resolve_kind(v: _Voltage, answers: dict[str, str]) -> MissingInput | None:
    """Fill ``v.kind`` from the mains answer when the words did not say; report a contradiction as a missing input."""
    mains = yes_no(answers.get(MAINS_KEY))
    if v.kind is None:
        if mains == "yes":
            v.kind, v.origin = "ac", v.origin + f"; AC because {MAINS_KEY}=yes"
        elif mains == "no":
            v.kind, v.origin = "dc", v.origin + f"; DC because {MAINS_KEY}=no"
        else:
            return MissingInput(key=MAINS_KEY, reason=f"{v.key} does not say AC or DC; answer {MAINS_KEY} (yes = AC mains, no = DC)")
        return None
    v.origin += f"; {v.kind.upper()} stated with the value"
    if mains is not None and (mains == "yes") != (v.kind == "ac"):
        said = "AC mains" if mains == "yes" else "DC"
        return MissingInput(key=MAINS_KEY, reason=f"{v.key} says {v.kind.upper()} ({v.origin.split(';')[0]}) but {MAINS_KEY}={mains} says {said}: "
                                                    f"the two contradict each other - correct the requirement or the answer")
    return None


def _format_v(volts: float | list[float]) -> str:
    if isinstance(volts, list):
        return f"{volts[0]:.12g}..{volts[1]:.12g} V"
    return f"{volts:.12g} V"


def _in_band(volts: float | list[float], band: list[float] | None) -> bool:
    if band is None:
        return False
    lo, hi = band
    if isinstance(volts, list):
        return volts[0] <= hi and volts[1] >= lo
    return lo <= volts <= hi


def _band_text(rule: ApplicabilityRule, kind: str) -> str:
    band = rule.ac if kind == "ac" else rule.dc
    return f"{band[0]:.12g}-{band[1]:.12g} V {kind.upper()}" if band is not None else f"no {kind.upper()} band (never in scope for {kind.upper()})"


def _voltage_rule(rule: ApplicabilityRule, answers: dict[str, str], requirements: Any) -> Evaluation:
    primary = rule.requirement or ""
    keys = [primary, *(k for k in rule.requirements if k != primary)]
    evidence = [rule.evidence] if rule.evidence else []
    voltages: list[_Voltage] = []
    missing: list[MissingInput] = []
    for i, key in enumerate(keys):
        v, m = _read_voltage(key, answers, requirements, required=(i == 0))
        if v is not None:
            voltages.append(v)
        if m is not None:
            missing.append(m)
    scope = rule.scope_answer
    scope_given = bool(scope and answers.get(scope, "").strip())
    if scope_given:
        assert scope is not None
        volts, why = _parse_text(answers[scope])
        if volts is None:
            missing.append(MissingInput(key=scope, reason=f"answer {scope}={answers[scope]!r} can not be read as a voltage: {why}"))
        else:
            voltages.append(_Voltage(key=scope, volts=volts, kind=_current_kind(answers[scope]), origin=f"answer {scope}"))
    inputs: dict[str, str] = {}
    decided: list[_Voltage] = []
    for v in voltages:
        m = _resolve_kind(v, answers)
        if m is not None:
            if m.key not in {x.key for x in missing}:
                missing.append(m)
            inputs[v.key] = f"{_format_v(v.volts)} ({v.origin})"
            continue
        inputs[v.key] = f"{_format_v(v.volts)} {(v.kind or '').upper()} ({v.origin})"
        decided.append(v)

    def part(v: _Voltage, inside: bool) -> str:
        return f"{v.key} = {_format_v(v.volts)} {(v.kind or '').upper()} ({v.origin}) is {'inside' if inside else 'outside'} the {_band_text(rule, v.kind or 'dc')} band"

    inside = [v for v in decided if _in_band(v.volts, rule.ac if v.kind == "ac" else rule.dc)]
    if inside:
        rationale = "; ".join(part(v, v in inside) for v in decided) + (f" stated in {rule.evidence}" if rule.evidence else "") + (f"; {rule.text}" if rule.text else "")
        return Evaluation(applicability=Applicability.APPLICABLE, status=ValidationStatus.NOT_VERIFIED, rationale=rationale, inputs_used=inputs, evidence=evidence)
    if missing or not decided:
        if not missing:
            missing.append(MissingInput(key=primary, reason=f"needs {primary}"))
        return Evaluation(applicability=Applicability.UNDECIDED, status=ValidationStatus.USER_INPUT_REQUIRED,
                          rationale="; ".join(m.reason for m in missing), inputs_used=inputs, missing=missing, evidence=evidence)
    if scope and not scope_given:
        assert scope is not None
        m = MissingInput(key=scope, reason=(f"every stated voltage ({', '.join(f'{v.key} = {_format_v(v.volts)} {(v.kind or '').upper()}' for v in decided)}) is outside the band, "
                                             f"but the rule rates the equipment by its highest voltage anywhere in the product (input, outputs, internal rails, isolation): "
                                             f"answer {scope} (e.g. '12 V DC')"))
        return Evaluation(applicability=Applicability.UNDECIDED, status=ValidationStatus.USER_INPUT_REQUIRED, rationale=m.reason, inputs_used=inputs, missing=[m], evidence=evidence)
    rationale = "; ".join(part(v, False) for v in decided) + (f" stated in {rule.evidence}" if rule.evidence else "") + (f"; {rule.text}" if rule.text else "")
    return Evaluation(applicability=Applicability.NOT_APPLICABLE, status=ValidationStatus.NOT_APPLICABLE, rationale=rationale, inputs_used=inputs, evidence=evidence)


# --------------------------------------------------------------------------- evaluation


def _combine_all(parts: list[Applicability]) -> Applicability:
    if any(p is Applicability.NOT_APPLICABLE for p in parts):
        return Applicability.NOT_APPLICABLE
    if all(p is Applicability.APPLICABLE for p in parts):
        return Applicability.APPLICABLE
    return Applicability.UNDECIDED


def _combine_any(parts: list[Applicability]) -> Applicability:
    if any(p is Applicability.APPLICABLE for p in parts):
        return Applicability.APPLICABLE
    if all(p is Applicability.NOT_APPLICABLE for p in parts):
        return Applicability.NOT_APPLICABLE
    return Applicability.UNDECIDED


def _evaluate(rule: ApplicabilityRule, answers: dict[str, str], requirements: Any) -> Evaluation:
    k = rule.kind
    if k == "always":
        return Evaluation(applicability=Applicability.APPLICABLE, status=ValidationStatus.NOT_VERIFIED,
                          rationale="applies by rule (always)" + (f": {rule.text}" if rule.text else ""), evidence=[rule.evidence] if rule.evidence else [])
    if k == "never":
        return Evaluation(applicability=Applicability.NOT_APPLICABLE, status=ValidationStatus.NOT_APPLICABLE,
                          rationale="excluded by rule (never)" + (f": {rule.text}" if rule.text else ""), evidence=[rule.evidence] if rule.evidence else [])
    if k == "answer":
        key = rule.key or ""
        wanted = [rule.equals] if isinstance(rule.equals, str) else list(rule.equals or [])
        raw = answers.get(key)
        if raw is None or not raw.strip():
            return Evaluation(applicability=Applicability.UNDECIDED, status=ValidationStatus.USER_INPUT_REQUIRED,
                              rationale=f"needs the answer to {key!r}", missing=[MissingInput(key=key, reason=f"scope question {key!r} not answered")],
                              evidence=[rule.evidence] if rule.evidence else [])
        wanted_yn = [yes_no(w) for w in wanted]
        if all(w is not None for w in wanted_yn):
            canon = yes_no(raw)
            if canon is None:
                return Evaluation(applicability=Applicability.UNDECIDED, status=ValidationStatus.USER_INPUT_REQUIRED,
                                  rationale=f"the answer to {key!r} is not yes/no",
                                  missing=[MissingInput(key=key, reason=f"answer {raw.strip()!r} to {key!r} is not yes or no; answer yes or no")],
                                  inputs_used={key: f"{raw.strip()} (answer, not understood)"}, evidence=[rule.evidence] if rule.evidence else [])
            holds = canon in wanted_yn
        else:
            canon = yes_no(raw) or normalise_answer(raw)
            wanted_canon = [yes_no(w) or normalise_answer(w) for w in wanted]
            holds = canon in wanted_canon
        applic = Applicability.APPLICABLE if holds else Applicability.NOT_APPLICABLE
        return Evaluation(applicability=applic, status=status_for(applic),
                          rationale=f"{key} = {raw.strip()!r} ({'is' if holds else 'is not'} {' / '.join(wanted)})" + (f"; {rule.text}" if rule.text else ""),
                          inputs_used={key: f"{raw.strip()} (answer)"}, evidence=[rule.evidence] if rule.evidence else [])
    if k == "voltage_range":
        return _voltage_rule(rule, answers, requirements)
    if k in ("all_of", "any_of"):
        parts = [_evaluate(r, answers, requirements) for r in rule.rules]
        applic = _combine_all([p.applicability for p in parts]) if k == "all_of" else _combine_any([p.applicability for p in parts])
        inputs: dict[str, str] = {}
        evidence: list[str] = []
        missing: list[MissingInput] = []
        for p in parts:
            inputs.update(p.inputs_used)
            evidence.extend(e for e in p.evidence if e not in evidence)
        if applic is Applicability.UNDECIDED:
            for p in parts:
                if p.applicability is Applicability.UNDECIDED:
                    missing.extend(m for m in p.missing if m.key not in {x.key for x in missing})
        joiner = " AND " if k == "all_of" else " OR "
        rationale = f"{k}[" + joiner.join(f"({p.rationale})" for p in parts) + f"] -> {applic.value}"
        if rule.text:
            rationale = f"{rule.text}: " + rationale
        if applic is not Applicability.UNDECIDED:
            # the decision rests only on the members that decided it; drop the evidence of undecided members
            evidence = []
            for p in parts:
                decisive = (k == "all_of" and p.applicability is Applicability.NOT_APPLICABLE) or (k == "any_of" and p.applicability is Applicability.APPLICABLE) \
                    or (k == "all_of" and applic is Applicability.APPLICABLE) or (k == "any_of" and applic is Applicability.NOT_APPLICABLE)
                if decisive:
                    evidence.extend(e for e in p.evidence if e not in evidence)
        return Evaluation(applicability=applic, status=status_for(applic), rationale=rationale, inputs_used=inputs, missing=missing, evidence=evidence)
    if k == "not":
        assert rule.rule is not None
        inner = _evaluate(rule.rule, answers, requirements)
        flipped = {Applicability.APPLICABLE: Applicability.NOT_APPLICABLE, Applicability.NOT_APPLICABLE: Applicability.APPLICABLE}.get(inner.applicability, Applicability.UNDECIDED)
        rationale = f"not({inner.rationale}) -> {flipped.value}" + (f"; {rule.text}" if rule.text else "")
        return Evaluation(applicability=flipped, status=status_for(flipped), rationale=rationale, inputs_used=inner.inputs_used, missing=inner.missing,
                          evidence=list(inner.evidence) + ([rule.evidence] if rule.evidence and rule.evidence not in inner.evidence else []))
    raise ValueError(f"unknown rule kind {k!r}")  # pragma: no cover - the model validator refuses it first


def evaluate(rule: ApplicabilityRule, answers: dict[str, str] | None, requirements: Any = None) -> Evaluation:
    """Decide ``rule`` from ``answers`` (scope-question key -> the user's text) and ``requirements`` (a :class:`~ai_eda.ir.RequirementSet`,
    a list of :class:`~ai_eda.ir.Requirement` or a ``key -> Requirement`` dict). See the module docstring."""
    return _evaluate(rule, dict(answers or {}), requirements)


__all__ = ["APPLICABILITY_VERSION", "MAINS_KEY", "TRUSTED_KINDS", "Evaluation", "MissingInput", "evaluate", "status_for", "yes_no"]
