"""Datasheet fact grounding: a claimed part characteristic enters the IR only when the archived datasheet says it verbatim.

Invariants (the same trust model as the requirement stage,
:mod:`ai_eda.llm.extraction`, whose helpers are reused rather than re-implemented):

* A :class:`DatasheetFact` is a *proposal* - from a model through
  :meth:`~ai_eda.llm.service.LLMService.structured` (strict schema,
  :func:`facts_json_schema`) or from a JSON file the user wrote
  (:func:`load_facts_file`). Who proposed it does not matter to grounding.
* :func:`ground_facts` accepts a fact only when the quote is found on the
  claimed page of the archived document with the archive's own
  :meth:`~ai_eda.tools.sources.ArchivedDocument.find_quote` (exact
  characters, whitespace free, token boundaries) after control characters
  and NBSP are mapped to spaces (:func:`searchable_document` - Infineon
  separates words with U+0002, ST text carries U+0000/U+0003; the mapping is
  length preserving so offsets stay exact). A **numeric** fact must in
  addition (1) re-read as one quantity from the document's own text at the
  hit *and* coincide with exactly one quantity :func:`~ai_eda.tools.calc.quantity.find_quantities`
  reads in the whole page there - the request stage's rule
  (:func:`~ai_eda.llm.extraction._request_quantity`, reused), so ``125 degC``
  cut out of ``-40 to 125 degC`` is a fragment of a range and is rejected;
  (2) carry the unit family its key demands (:func:`expected_unit`:
  ``v_*`` volts, ``i_*`` amperes, ``power_rating`` watts, ``tolerance``
  percent, ``operating_temperature`` degC, ...) - a key with no known family
  can not be checked and is rejected; (3) agree with the parse in the
  proposer's ``value``/``unit`` (relative tolerance
  :data:`~ai_eda.llm.extraction.REL_TOL`, canonical unit: ``100 mW`` agrees
  with ``0.1 W``). A **text** fact must have its value verbatim inside the
  quote; ``package`` must in addition quote the row of *this* part - the
  component's MPN must stand inside the quote (token boundary, ASCII case
  ignored), else an ordering table's other row would name the wrong package.
  ``manufacturer`` is a document-level fact and needs no MPN. Anything else
  is rejected with the reason and never enters the IR.
* What grounding can not check is the **meaning** a proposer attached to a
  number - that ``0.6 V`` is a maximum rating and not a dropout voltage is
  decided by whoever chose the key. A fact the *user* wrote is therefore
  ``authoritative`` at once (the user read the page); a fact a *model*
  proposed is only shown, under the required question ``confirm_facts`` of
  :class:`~ai_eda.agents.component.ComponentAgent`, and enters the IR only
  when the user confirms that table in a later run (``confirmed_by`` marks
  the note). Until then nothing of it is in the IR.
* An accepted fact becomes an ``authoritative`` :class:`~ai_eda.ir.Traced`
  whose :class:`~ai_eda.ir.SourceRef` names the archived document by hash and
  the page (``section="page N"``) and whose note keeps the quote as it stands
  in the document, the deterministic parse and the extractor stamp
  (:attr:`~ai_eda.tools.sources.ArchivedDocument.extraction_stamp`: extractor,
  version, text hash - so an extractor upgrade that changes the text is
  visible against the claim). :func:`apply_facts` writes it to
  ``Component.manufacturer`` / ``Component.package`` / ``Component.electrical[key]``.
* Identity and structure are not facts: ``mpn`` (established by the existence
  check, :mod:`ai_eda.parts.existence`), ``ref``, ``value``, ``datasheet``,
  ``symbol``, ``footprint``, ``pins``, ``sourcing`` are :data:`RESERVED_KEYS`
  and always rejected. Items carrying a directive phrase
  (:func:`~ai_eda.llm.extraction.find_directive`) are rejected: document text
  and model output are data.
"""

from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ai_eda.ir import Component, Evidence, Traced, ValidationResult, ValidationStatus, authoritative
from ai_eda.llm.client import LLMMessage, LLMResponse
from ai_eda.llm.extraction import (  # shared grounding helpers, reused on purpose (see the module docstring)
    REL_TOL,
    _mismatch,
    _model_quantity,
    _request_quantity,
    _strictify,
    _traced_payload,
    canonical_key,
    find_directive,
    find_quote,
)
from ai_eda.llm.prompts import datasheet_fact_messages
from ai_eda.llm.router import TaskKind
from ai_eda.llm.service import LLMService
from ai_eda.tools.calc.quantity import format_quantity
from ai_eda.tools.sources import ArchivedDocument

#: bumped when the grounding rules or the schema change
FACTS_VERSION = "0.2"
TOOL = "parts.datasheet_facts"
CHECK_PREFIX = "component.facts."

#: keys a fact may never set: identity comes from the existence check, structure from the design
RESERVED_KEYS: frozenset[str] = frozenset({"mpn", "ref", "value", "datasheet", "symbol", "footprint", "pins", "sourcing", "spice", "provenance", "description"})
#: text facts that target a ``Component`` field rather than ``electrical``
IDENTITY_KEYS: frozenset[str] = frozenset({"manufacturer", "package"})
#: keys a model is asked for by default (any other canonical key with a known unit family is accepted too)
DEFAULT_FACT_KEYS: tuple[str, ...] = ("manufacturer", "package", "v_max", "i_max", "power_rating", "tolerance", "operating_temperature")

#: canonical unit (of :mod:`ai_eda.tools.calc.quantity`) a numeric fact key must carry: exact keys, then prefixes, then suffixes
KEY_UNITS: dict[str, str] = {
    "power_rating": "W", "tolerance": "percent", "operating_temperature": "degC", "storage_temperature": "degC", "junction_temperature": "degC",
    "resistance": "ohm", "capacitance": "F", "inductance": "H", "frequency": "Hz", "voltage": "V", "current": "A", "power": "W", "temperature": "degC",
    "esr": "ohm", "rds_on": "ohm", "quiescent_current": "A", "dropout_voltage": "V",
}
KEY_PREFIX_UNITS: dict[str, str] = {"v_": "V", "i_": "A", "p_": "W", "r_": "ohm", "c_": "F", "l_": "H", "f_": "Hz", "t_": "degC"}
KEY_SUFFIX_UNITS: dict[str, str] = {
    "_voltage": "V", "_current": "A", "_power": "W", "_temperature": "degC", "_resistance": "ohm", "_capacitance": "F", "_inductance": "H",
    "_frequency": "Hz", "_tolerance": "percent", "_percent": "percent",
}

#: ASCII-only lower-casing (length preserving) for MPN matching inside a quote
_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def expected_unit(key: str) -> str | None:
    """The canonical unit family ``key`` demands (``v_max`` -> ``V``, ``power_rating`` -> ``W``), or ``None`` when none is known."""
    if key in KEY_UNITS:
        return KEY_UNITS[key]
    for prefix, unit in KEY_PREFIX_UNITS.items():
        if key.startswith(prefix):
            return unit
    for suffix, unit in KEY_SUFFIX_UNITS.items():
        if key.endswith(suffix):
            return unit
    return None


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasheetFact(_Strict):
    """One claim about a part, to be grounded in the archived datasheet."""

    key: str = Field(description="ascii lower-case snake_case: manufacturer, package, or an electrical characteristic (v_max, i_max, power_rating, tolerance, operating_temperature, ...)")
    value: float | str = Field(description="the number as written in the quote (200 for '200 V'; the low end of a range), or the text value for manufacturer / package")
    value_high: float | None = Field(default=None, description="the high end when the quote states a range ('-40 to 125 degC': value -40, value_high 125); null otherwise")
    unit: str | None = Field(description="the unit as written in the quote (V, mA, W, %, degC); null for a text value")
    page: int = Field(description="1-based page of the datasheet the quote stands on")
    quote: str = Field(description="verbatim phrase from that page containing exactly this value (or range) and its unit")


class DatasheetFacts(_Strict):
    facts: list[DatasheetFact]
    not_found: list[str] = Field(description="keys asked for that the datasheet text does not state")


def facts_json_schema() -> dict[str, Any]:
    """The strict JSON schema for :class:`DatasheetFacts` (every object closed and fully required)."""
    return _strictify(DatasheetFacts.model_json_schema())


# --------------------------------------------------------------------------- document text

#: C0/C1 control characters (except tab / newline / carriage return) and NBSP -> space, length preserving
_CONTROL_TO_SPACE: dict[int, int] = {c: 0x20 for c in range(0x00, 0x20) if chr(c) not in "\t\n\r"}
_CONTROL_TO_SPACE.update({c: 0x20 for c in range(0x7F, 0xA0)})
_CONTROL_TO_SPACE[0xA0] = 0x20


def clean_control_chars(text: str) -> str:
    """``text`` with control characters and NBSP replaced by spaces (same length, so offsets into the original hold)."""
    return text.translate(_CONTROL_TO_SPACE)


def searchable_document(doc: ArchivedDocument) -> ArchivedDocument:
    """A copy of ``doc`` whose pages have control characters mapped to spaces - what quotes are grounded against."""
    return doc.model_copy(update={"pages": [clean_control_chars(p) for p in doc.pages]})


# --------------------------------------------------------------------------- grounding


class AcceptedFact(BaseModel):
    key: str
    #: "manufacturer" | "package" | "electrical"
    target: str
    traced: Traced
    page: int
    #: the document's own text at the hit
    quote: str
    context: str
    #: the deterministic parse of the quote (numeric facts) or "(text)"
    parsed: str


class GroundedFacts(BaseModel):
    document: str  # sha256 of the archived document
    accepted: list[AcceptedFact] = Field(default_factory=list)
    #: (key, reason) - never enters the IR
    rejected: list[tuple[str, str]] = Field(default_factory=list)
    proposer: str = "user"
    #: the extractor stamp of the text the facts were grounded against
    extraction: str = ""

    @property
    def accepted_keys(self) -> list[str]:
        return [a.key for a in self.accepted]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _quantity_at(page_text: str, span: tuple[int, int]) -> tuple[Any, str | None]:
    """``(parsed, None)`` for the one quantity the page states at ``span``, else ``(None, reason)``.

    The request-side rule, reused: the quote must parse as one quantity and
    coincide with exactly one of the quantities ``find_quantities`` reads in
    the whole page - a fragment of a range or a tolerance (``125 degC`` out
    of ``-40 to 125 degC``) is not what the document states.
    """
    parsed, reason = _request_quantity(page_text, span)
    if reason is not None:
        return None, reason.replace("in the request", "on the page")
    return parsed, None


def fact_row(fact: AcceptedFact) -> dict[str, Any]:
    """What a person is shown (and later confirms) about an accepted fact: key, target, value, unit, page, quote, parse."""
    return {"key": fact.key, "target": fact.target, "value": fact.traced.value, "unit": fact.traced.unit, "page": fact.page, "quote": fact.quote, "parsed": fact.parsed}


def ground_facts(
    doc: ArchivedDocument,
    facts: list[DatasheetFact] | DatasheetFacts,
    *,
    proposer: str = "user",
    title: str | None = None,
    authority: str | None = None,
    mpn: str | None = None,
    confirmed_by: str | None = None,
) -> GroundedFacts:
    """Check every fact against ``doc`` (module docstring); accepted facts carry ``authoritative`` provenance to the document page.

    ``mpn`` is the component's part number a ``package`` fact must quote;
    ``confirmed_by`` names the answer key the user confirmed a model's table
    with (appended to the note) - the caller passes it only after that
    confirmation.
    """
    items = facts.facts if isinstance(facts, DatasheetFacts) else list(facts)
    sdoc = searchable_document(doc)
    out = GroundedFacts(document=doc.sha256, proposer=proposer, extraction=doc.extraction_stamp)
    seen: set[str] = set()
    tail = f"; proposed by {proposer}; {doc.extraction_stamp}" + (f"; confirmed by user ({confirmed_by})" if confirmed_by else "")
    for fact in items:
        key = canonical_key(fact.key)
        if key is None:
            out.rejected.append((fact.key, "key has no ascii letters after canonicalisation"))
            continue
        if key in RESERVED_KEYS:
            out.rejected.append((key, "identity / structure keys are never set by a fact (the MPN is established by the existence check)"))
            continue
        hit_phrase = find_directive(fact.key, fact.quote, str(fact.value), fact.unit)
        if hit_phrase is not None:
            out.rejected.append((key, f"directive phrase {hit_phrase!r} in the fact; document text and model output are data, not instructions"))
            continue
        if key in seen:
            out.rejected.append((key, "duplicate key; the first fact was kept"))
            continue
        if not fact.quote.strip():
            out.rejected.append((key, "empty quote"))
            continue
        if fact.page < 1 or fact.page > doc.page_count:
            out.rejected.append((key, f"page {fact.page} does not exist (the archived document has {doc.page_count} page(s))"))
            continue
        hits = sdoc.find_quote(fact.quote, fact.page)
        if not hits:
            elsewhere = [h.page for h in sdoc.find_quote(fact.quote)]
            where = f" (found on page(s) {elsewhere} instead)" if elsewhere else ""
            out.rejected.append((key, f"quote not found verbatim on page {fact.page}: {fact.quote!r}{where}"))
            continue
        hit = hits[0]
        numeric = _is_number(fact.value)
        if key in IDENTITY_KEYS and numeric:
            out.rejected.append((key, f"{key} is a text fact; got the number {float(fact.value):g}"))
            continue
        if key not in IDENTITY_KEYS and not numeric:
            out.rejected.append((key, f"electrical facts must be numeric with a unit; got the text {fact.value!r}"))
            continue
        source = doc.source_ref(title=title, section=hit.section, authority=authority)
        if numeric:
            page_text = sdoc.pages[hit.page - 1]
            parsed, reason = _quantity_at(page_text, (hit.offset, hit.offset + len(hit.matched)))
            if parsed is None:
                out.rejected.append((key, str(reason)))
                continue
            model_q = _model_quantity(float(fact.value), fact.unit, fact.value_high)
            if model_q is None:
                if fact.value_high is not None:
                    out.rejected.append((key, f"unit not recognised or range not low..high: {fact.unit!r} (value {fact.value!r}..{fact.value_high!r})"))
                else:
                    out.rejected.append((key, f"unit not recognised: {fact.unit!r} (value {fact.value!r})"))
                continue
            family = expected_unit(key)
            if family is None:
                out.rejected.append((key, f"no unit family is known for key {key!r}, so the unit can not be checked against the key "
                                          f"(use a v_ / i_ / p_ / r_ / c_ / l_ / f_ / t_ key or one of {sorted(KEY_UNITS)})"))
                continue
            if model_q.unit != family:
                out.rejected.append((key, f"unit mismatch: key {key!r} expects {family}, got {fact.unit!r} ({format_quantity(model_q)})"))
                continue
            reason = _mismatch(model_q, parsed)
            if reason is not None:
                out.rejected.append((key, reason))
                continue
            value, unit = _traced_payload(parsed)
            parsed_text = format_quantity(parsed)
            note = f"quote: {hit.matched!r}; parsed: {parsed_text}; page {hit.page}{tail}"
            traced = authoritative(value, source, unit=unit, note=note)
            target = "electrical"
        else:
            text_value = str(fact.value).strip()
            if fact.unit:
                out.rejected.append((key, f"a text value has no unit; got {fact.unit!r}"))
                continue
            if not text_value or find_quote(text_value, hit.matched) is None:
                out.rejected.append((key, f"value {text_value!r} not found verbatim inside the quote {hit.matched!r}"))
                continue
            if key == "package":
                # the row of this part: an ordering table lists one package per orderable code
                if not mpn or not str(mpn).strip():
                    out.rejected.append((key, "package is not tied to the part: the component has no MPN to find in the quoted row"))
                    continue
                if find_quote(str(mpn).translate(_ASCII_LOWER), hit.matched.translate(_ASCII_LOWER)) is None:
                    out.rejected.append((key, f"package is not tied to the part: the quote {hit.matched!r} does not contain the MPN {mpn!r} "
                                              "(quote the ordering row that names this part number and its package)"))
                    continue
            parsed_text = "(text)"
            note = f"quote: {hit.matched!r}; page {hit.page}{tail}"
            traced = authoritative(text_value, source, note=note)
            target = key
        seen.add(key)
        out.accepted.append(AcceptedFact(key=key, target=target, traced=traced, page=hit.page, quote=hit.matched, context=hit.context, parsed=parsed_text))
    return out


def apply_facts(component: Component, accepted: list[AcceptedFact]) -> Component:
    """A copy of ``component`` with the accepted facts written to ``manufacturer`` / ``package`` / ``electrical`` (the input is not mutated)."""
    c = component.model_copy(deep=True)
    for fact in accepted:
        if fact.target == "manufacturer":
            c.manufacturer = fact.traced
        elif fact.target == "package":
            c.package = fact.traced
        else:
            c.electrical[fact.key] = fact.traced
    return c


def facts_result(
    component_ref: str, doc: ArchivedDocument, grounded: GroundedFacts, *, check_id: str | None = None,
    status: ValidationStatus | None = None, message_suffix: str | None = None,
) -> ValidationResult:
    """``component.facts.<ref>`` (or ``check_id``): PASS when every proposed fact was grounded, NOT_VERIFIED when any was rejected or none was proposed.

    ``status`` overrides that rule (the agent passes ``USER_INPUT_REQUIRED``
    while a model's table awaits confirmation); ``message_suffix`` is appended.
    """
    n_ok, n_bad = len(grounded.accepted), len(grounded.rejected)
    if status is None:
        status = ValidationStatus.PASS if n_ok and not n_bad else ValidationStatus.NOT_VERIFIED
    message = f"{n_ok} fact(s) grounded in the archived datasheet, {n_bad} rejected (proposed by {grounded.proposer})"
    if grounded.rejected:
        message += ": " + "; ".join(f"{k}: {why}" for k, why in grounded.rejected[:3]) + (" ..." if n_bad > 3 else "")
    if message_suffix:
        message += f"; {message_suffix}"
    return ValidationResult(
        check_id=check_id or f"{CHECK_PREFIX}{component_ref}",
        status=status,
        message=message,
        tool=TOOL,
        tool_version=FACTS_VERSION,
        artifact_hash=doc.sha256,
        evidence=[Evidence(description=f"archived datasheet grounding the facts of {component_ref}", path=str(doc.path), url=doc.final_url or doc.url, content_hash=doc.sha256)],
        details={
            "ref": component_ref,
            "document": doc.sha256,
            "proposer": grounded.proposer,
            "accepted": [{**fact_row(a), "context": a.context} for a in grounded.accepted],
            "rejected": [{"key": k, "reason": why} for k, why in grounded.rejected],
            "rel_tol": REL_TOL,
            "extraction": grounded.extraction,
            "extractor": doc.extractor, "extractor_version": doc.extractor_version, "library_version": doc.library_version,
            "text_sha256": doc.text_sha256, "text_matches_meta": doc.text_matches_meta,
        },
    )


# --------------------------------------------------------------------------- proposals


def load_facts_file(path: Path | str) -> tuple[dict[str, list[DatasheetFact]], list[str]]:
    """Facts the user wrote as JSON, by reference designator, plus the entries that do not validate (as messages).

    Two layouts are accepted: ``{"R1": [fact, ...], "U1": [...]}`` and
    ``{"facts": [{"ref": "R1", ...fact}, ...]}`` (or a bare list of the
    latter). Each fact is validated against the strict :class:`DatasheetFact`
    schema; an invalid entry is reported and skipped, never guessed at.
    """
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return {}, [f"facts file {p} unreadable: {e}"]
    out: dict[str, list[DatasheetFact]] = {}
    errors: list[str] = []

    def add(ref: Any, item: Any, where: str) -> None:
        if not isinstance(ref, str) or not ref.strip():
            errors.append(f"{where}: missing reference designator")
            return
        if not isinstance(item, dict):
            errors.append(f"{where}: not an object")
            return
        try:
            fact = DatasheetFact.model_validate(item)
        except ValidationError as e:
            errors.append(f"{where}: {' '.join(str(e).split())[:300]}")
            return
        out.setdefault(ref.strip(), []).append(fact)

    items: Any
    if isinstance(data, dict) and "facts" in data:
        items = data["facts"]
    elif isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for ref, facts in data.items():
            if not isinstance(facts, list):
                errors.append(f"{ref}: expected a list of facts")
                continue
            for i, item in enumerate(facts):
                add(ref, item, f"{ref}[{i}]")
        return out, errors
    else:
        return {}, [f"facts file {p}: expected an object or a list"]
    if not isinstance(items, list):
        return {}, [f"facts file {p}: 'facts' must be a list"]
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            errors.append(f"facts[{i}]: not an object")
            continue
        body = {k: v for k, v in item.items() if k != "ref"}
        add(item.get("ref"), body, f"facts[{i}]")
    return out, errors


def llm_fact_proposals(
    llm: LLMService,
    doc: ArchivedDocument,
    component: Component,
    keys: tuple[str, ...] | list[str] = DEFAULT_FACT_KEYS,
) -> tuple[DatasheetFacts, LLMResponse, list[LLMMessage]]:
    """Ask a model for facts about ``component`` from the archived document's text (a proposal; ground it with :func:`ground_facts`).

    The prompt carries the extracted pages with page markers; the model's
    answer is validated against the strict schema by the service. Raises what
    the service raises (``LLMError``, ``BudgetExceededError``,
    ``StructuredOutputError``).
    """
    about = {
        "ref": component.ref, "value": component.value, "description": component.description,
        "mpn": component.mpn.value if component.mpn is not None else None,
        "manufacturer": component.manufacturer.value if component.manufacturer is not None else None,
    }
    messages = datasheet_fact_messages(about, list(doc.pages), list(keys))
    facts, resp = llm.structured(TaskKind.COMPONENT_PROPOSAL, messages, DatasheetFacts, json_schema=facts_json_schema())
    return facts, resp, messages


__all__ = [
    "CHECK_PREFIX",
    "DEFAULT_FACT_KEYS",
    "FACTS_VERSION",
    "IDENTITY_KEYS",
    "KEY_PREFIX_UNITS",
    "KEY_SUFFIX_UNITS",
    "KEY_UNITS",
    "RESERVED_KEYS",
    "TOOL",
    "AcceptedFact",
    "DatasheetFact",
    "DatasheetFacts",
    "GroundedFacts",
    "apply_facts",
    "clean_control_chars",
    "expected_unit",
    "fact_row",
    "facts_json_schema",
    "facts_result",
    "ground_facts",
    "llm_fact_proposals",
    "load_facts_file",
    "searchable_document",
]
