"""Grounding of LLM requirement extractions - pure functions, no I/O, no model call.

Invariant: nothing a model returns enters the IR as authoritative. A model's
output is parsed into the strict schemas below (:func:`json_schema` is
OpenAI/OpenRouter strict-mode compatible: every object has
``additionalProperties: false`` and lists every property as required), and
:func:`ground_extraction` then checks every claim deterministically before it
becomes a :class:`~ai_eda.ir.Requirement`:

* An **explicit** requirement must quote the user's request verbatim.
  :func:`find_quote` locates the quote in the request character by character
  (whitespace may differ, nothing else - the SI prefix ``m`` is not ``M``)
  and only at a *token boundary*: the character before the match may not be
  a digit, a sign, a decimal/thousands mark or an ASCII letter, and the
  character after it may not be an ASCII letter or digit (Hangul may follow,
  Korean particles attach without a space). So ``5V`` is not found inside
  ``-5V``, ``0.5A``, ``12V`` or ``5Vin``. The numeric value is then parsed
  from the **request's own text at that span** (never from the model's copy)
  by the deterministic quantity parser (:mod:`ai_eda.tools.calc.quantity`),
  and it must be exactly the quantity :func:`~ai_eda.tools.calc.quantity.find_quantities`
  reads at that place in the whole request (a quote that is one bound of a
  range, ``5 V`` in ``3.3 to 5 V``, is not the user's statement). The model's
  own ``number``/``unit`` must agree with that parse (relative tolerance
  :data:`REL_TOL`, unit in canonical form; ``500 mA`` and ``0.5 A`` both
  agree with the quote ``500mA``; a ``±5 %`` quote agrees with ``5 %`` or
  ``-5..5 %`` and enters as the symmetric range ``[-5, 5]``). Anything that
  fails is **demoted** to an assumption (``kind=assumption``,
  ``status=assumed``, ``assumption`` provenance) with the reason in the
  provenance note and in :attr:`GroundedExtraction.demoted` - it is never
  silently accepted and never silently lost. A grounded explicit value
  carries ``llm_generated`` provenance (``tool`` = model, note = the quote as
  it stands in the request + the deterministic parse, prefixed
  :data:`GROUNDED_NOTE_PREFIX`) until the user confirms it, when
  :func:`upgrade_confirmed` re-tags it ``user_requirement`` (the note keeps
  the model and the quote). Its human-readable ``text`` is built from the
  user's words (``<key>: <quote as written>``); the model's one-sentence
  statement is kept in the note only, so ungrounded prose never becomes the
  statement of a confirmed requirement.
* What is **not** checked: which ``key`` and ``category`` the model attached
  to a quote. ``12V`` under ``output_voltage`` grounds as 12 V; only the
  confirmation table (which shows every quote in its request context) lets
  the user catch a wrong assignment.
* An **implicit** requirement needs a rationale (dropped otherwise); a quote
  on it is ignored (kept in the note). It stays ``llm_generated``.
* A model **assumption** needs a rationale (dropped otherwise) and enters
  with ``assumption`` provenance, which
  :class:`~ai_eda.validation.structural.AssumptionsSurfacedValidator` turns
  into ``USER_INPUT_REQUIRED``.
* **Keys** are canonicalised to ascii lower-case snake_case
  (:func:`canonical_key`); a key with no ascii letter left is dropped.
  Duplicate keys are merged when their values are identical, merged (text
  joined, the one numeric value kept) when they are of the *same kind* and at
  most one has a numeric value, turned into a
  :class:`~ai_eda.ir.RequirementConflict` (both marked ``conflicting``) when
  two numeric values differ, and otherwise **kept apart**: a value never moves
  between an explicit statement and an implicit / assumed item (the lower-
  ranked item gets the id ``req.<key>.<kind>``), so a model's number can not
  ride an explicit item into ``user_requirement`` on confirmation.
* **Questions** need a canonical snake_case key; the reserved
  :data:`CONFIRM_KEY` and keys already answered by a grounded explicit
  requirement are dropped, duplicates keep the first. They enter with
  ``source="llm"`` and are capped at :data:`MAX_QUESTION_CHARS`: a
  model-authored question is text addressed to the human and is labelled as
  such wherever it is shown.
* **Jurisdictions** and the **application** are taken only from a verbatim
  quote; jurisdiction codes must be 2-3 upper-case letters (``EU``, ``US``,
  ``KR``) *and* the quote must name that jurisdiction (the code itself as an
  upper-case token, or a name from :data:`JURISDICTION_NAMES`): ``EU`` is
  not grounded on ``US에서 판매``.
* **Prompt injection**: the request is data. Any extracted item whose text
  (key, text, quote, rationale, question, description, summary) contains one
  of :data:`DIRECTIVE_PHRASES` - phrases addressed to an assistant, matched
  case-insensitively after collapsing whitespace - is dropped with a note.
  The list is defence in depth, not the defence: nothing the model returns
  is ever executed, the schema has no free-form "instructions" field, and
  model-authored text reaching the human is labelled.
* The extraction **cache** is keyed by the request text; an entry also
  records :func:`extraction_fingerprint` (extraction version, prompt hash,
  schema hash) and :func:`cache_entry_staleness` tells an agent when an
  entry was produced under different rules and must be re-extracted rather
  than replayed.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ai_eda.ir.provenance import Provenance, ProvenanceKind, Traced
from ai_eda.ir.requirements import (
    MissingInformation,
    Requirement,
    RequirementConflict,
    RequirementKind,
    RequirementStatus,
)
from ai_eda.llm.client import LLMMessage
from ai_eda.llm.prompts import JSON_ONLY_INSTRUCTION, REQUIREMENT_EXTRACTION_SYSTEM, requirement_extraction_messages  # noqa: F401
from ai_eda.tools.calc.quantity import QUANTITY_VERSION, Quantity, QuantityRange, find_quantities, format_quantity, parse_quantity

#: bumped whenever grounding rules, the prompt or the schema change in a way that makes a cached model reply stale
EXTRACTION_VERSION = "0.2"

#: the one question the agent always asks after an extraction; the user's answer decides the upgrade
CONFIRM_KEY = "confirm_requirements"
#: answer keys that decide model-inferred items one by one (comma-separated requirement keys); see :func:`decide_inferred`
ACCEPT_KEY = "accept_implicit"
REJECT_KEY = "reject_implicit"
#: answers (after :func:`normalise_answer`) that confirm every grounded explicit item
CONFIRM_ANSWERS: frozenset[str] = frozenset({
    "yes", "y", "yep", "yeah", "yes please", "confirm", "confirmed", "ok", "okay", "correct",
    "네", "예", "응", "확인", "확인합니다", "맞습니다", "맞아요", "맞음", "좋아요", "승인",
})
#: answers that reject the table without saying what is wrong - the agent asks again instead of re-extracting
NEGATIVE_ANSWERS: frozenset[str] = frozenset({"no", "n", "nope", "cancel", "아니오", "아니요", "아니", "취소", "틀림", "틀렸어요", "틀렸습니다"})
#: a non-confirming answer shorter than this (after normalisation), or a single word without a digit, is not a
#: correction: the agent asks again
MIN_CORRECTION_CHARS = 4
#: relative tolerance between the model's number and the deterministic parse of its quote
REL_TOL = 1e-9
#: how :func:`upgrade_confirmed` marks a value's note; an agent recognises confirmed extraction items by it
CONFIRMED_NOTE_PREFIX = "confirmed by user"
#: how :func:`_ground_explicit` marks a value's note; :func:`is_grounded_explicit` requires it
GROUNDED_NOTE_PREFIX = "quote: "
_GROUNDED_NOTE_MARKER = "; parsed: "
#: longest model-authored question kept (characters); longer ones are cut and marked
MAX_QUESTION_CHARS = 500
#: characters of request text shown on each side of a quote in the confirmation table
CONTEXT_CHARS = 20

#: Phrases addressed to an assistant. An extracted item containing one (case-insensitive, whitespace
#: collapsed) is dropped: the request is data, and text that talks to the model is not a requirement.
#: Deliberately narrow - a requirement can legitimately say "must", "ignore noise", "act as a buffer" or
#: "power system:", so the list holds assistant-directed collocations only ("you must" is one: a request
#: that addresses the assistant is not stating a requirement).
DIRECTIVE_PHRASES: tuple[str, ...] = (
    # English
    "ignore previous",
    "ignore all previous",
    "ignore the above",
    "ignore prior",
    "ignore your",
    "disregard previous",
    "disregard the above",
    "disregard all",
    "disregard your",
    "prior instructions",
    "previous instructions",
    "instructions above",
    "everything above",
    "system prompt",
    "developer message",
    "as an ai",
    "as a language model",
    "you must",
    "you are now",
    "you should now",
    "new instructions",
    "new instruction:",
    "override your",
    "forget your",
    "reveal your",
    "print your",
    "output your",
    "repeat your",
    "reply with your",
    "respond with your",
    "api key",
    "api_key",
    "your key",
    "secret key",
    "pretend to be",
    "jailbreak",
    "assistant:",
    "<|im_start|>",
    "### instruction",
    # Korean
    "이전 지시",
    "이전 명령",
    "이전 안내",
    "앞의 지시",
    "위의 지시",
    "위 지시",
    "지시를 무시",
    "지시 무시",
    "명령을 무시",
    "안내는 잊",
    "지시는 잊",
    "명령은 잊",
    "시스템 프롬프트",
    "당신은 이제",
    "너는 이제",
    "ai로서",
    "인공지능으로서",
    "역할을 바꿔",
    "api 키",
    "키를 알려",
    "키를 출력",
    "비밀번호",
)

#: names by which a request may refer to a jurisdiction code (ASCII names matched case-insensitively as
#: words, others as substrings); the code itself as an upper-case token is always accepted
JURISDICTION_NAMES: dict[str, tuple[str, ...]] = {
    "EU": ("europe", "european union", "유럽", "유럽연합"),
    "US": ("usa", "u.s.", "u.s.a.", "united states", "미국"),
    "KR": ("korea", "south korea", "republic of korea", "한국", "대한민국"),
    "JP": ("japan", "일본"),
    "CN": ("china", "prc", "중국"),
    "GB": ("uk", "u.k.", "united kingdom", "britain", "great britain", "영국"),
    "DE": ("germany", "deutschland", "독일"),
    "FR": ("france", "프랑스"),
    "CA": ("canada", "캐나다"),
    "AU": ("australia", "호주"),
    "TW": ("taiwan", "대만"),
    "IN": ("india", "인도"),
}

Category = Literal["electrical", "thermal", "mechanical", "environmental", "regulatory", "safety", "application", "other"]


class _Strict(BaseModel):
    """Extra keys are rejected on validation and the emitted schema forbids them."""

    model_config = ConfigDict(extra="forbid")


class ExtractedValue(_Strict):
    quote: str = Field(description="verbatim phrase of the request that contains this number and its unit")
    number: float = Field(allow_inf_nan=False, description="the number as written in the quote (500 for '500mA', or 0.5 with unit 'A')")
    unit: str = Field(description="the unit as written (V, mA, kΩ, uF, MHz, °C, %); SI symbols with SI prefixes")
    number_high: float | None = Field(allow_inf_nan=False, description="upper bound when the quote states a range (-20..85 °C); null otherwise")


class ExtractedRequirement(_Strict):
    key: str = Field(description="ascii lower-case snake_case identifier, e.g. output_voltage, efficiency")
    text: str = Field(description="one-sentence statement of the requirement")
    kind: Literal["explicit", "implicit"] = Field(description="explicit: stated in the request; implicit: follows from it")
    category: Category
    quote: str | None = Field(description="explicit: verbatim phrase of the request stating it; implicit: null")
    value: ExtractedValue | None = Field(description="the numeric value with its quote, or null when the requirement has none")
    rationale: str | None = Field(description="implicit: why it follows from the request; explicit: null")


class ExtractedQuestion(_Strict):
    key: str = Field(description="ascii lower-case snake_case identifier for the missing information")
    question: str
    required: bool = Field(description="true when the design cannot proceed without the answer")
    options: list[str] = Field(description="choices to offer; empty for free-form")
    rationale: str | None


class ExtractedConflict(_Strict):
    keys: list[str] = Field(description="keys of the requirements that contradict each other")
    description: str


class ExtractedAssumption(_Strict):
    key: str = Field(description="ascii lower-case snake_case identifier")
    text: str
    category: Category
    number: float | None = Field(allow_inf_nan=False)
    unit: str | None
    number_high: float | None = Field(allow_inf_nan=False, description="upper bound when the assumption is a range; null otherwise")
    rationale: str = Field(description="why this had to be assumed and why this value")


class ExtractedApplication(_Strict):
    summary: str = Field(description="the intended application / use in a few words")
    quote: str = Field(description="verbatim phrase of the request stating it")


class ExtractedJurisdiction(_Strict):
    code: str = Field(description="ISO 3166 alpha-2 code or EU")
    quote: str = Field(description="verbatim phrase of the request naming the market")


class RequirementExtraction(_Strict):
    requirements: list[ExtractedRequirement]
    questions: list[ExtractedQuestion]
    conflicts: list[ExtractedConflict]
    assumptions: list[ExtractedAssumption]
    application: ExtractedApplication | None
    jurisdictions: list[ExtractedJurisdiction]


def _strictify(node: Any) -> Any:
    """Make a pydantic JSON schema strict-mode compatible in place: no defaults, every object closed and fully required."""
    if isinstance(node, dict):
        node.pop("default", None)
        if "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"])
        for v in node.values():
            _strictify(v)
    elif isinstance(node, list):
        for v in node:
            _strictify(v)
    return node


def json_schema() -> dict[str, Any]:
    """The strict JSON schema for :class:`RequirementExtraction` (``$defs`` inline, no defaults)."""
    return _strictify(RequirementExtraction.model_json_schema())


# --- text helpers -------------------------------------------------------------


def _squash(s: str) -> str:
    return "".join(s.split()).casefold()


def _collapse(s: str) -> str:
    return " ".join(s.split()).casefold()


#: what may not precede a quote match: it would continue a number (sign, digit, decimal / thousands mark,
#: tolerance mark) or an ASCII word / identifier
_NO_BEFORE = frozenset("0123456789.,+-−±_")


def _boundary_before(ch: str) -> bool:
    return not ch or not (ch in _NO_BEFORE or (ch.isascii() and ch.isalpha()))


def _boundary_after(ch: str) -> bool:
    return not ch or not (ch.isascii() and (ch.isalnum() or ch == "_"))


#: characters that continue an orderable code (``LM2596S-5.0``, ``VR1-0603-200V-A``): in identifier mode a match may not
#: stop or start next to one of them when an ASCII letter / digit continues on the other side. A slash separates an
#: option suffix (``LM2596S-5.0/NOPB``), so the code before it is a whole identifier.
_IDENTIFIER_JOINERS = frozenset(".-_")


def _identifier_boundary(text: str, s: int, e: int) -> bool:
    """Whether ``text[s:e]`` is a whole identifier: not cut out of a longer dot / dash / slash-joined code."""
    after = text[e] if e < len(text) else ""
    after2 = text[e + 1] if e + 1 < len(text) else ""
    if after in _IDENTIFIER_JOINERS and after2.isascii() and after2.isalnum():
        return False
    before = text[s - 1] if s > 0 else ""
    before2 = text[s - 2] if s > 1 else ""
    if before in _IDENTIFIER_JOINERS and before2.isascii() and before2.isalnum():
        return False
    return True


def find_quote(quote: str | None, raw_input: str, *, identifier: bool = False) -> tuple[int, int] | None:
    """``(start, end)`` of the first place ``quote`` occurs in ``raw_input`` at a token boundary, else ``None``.

    Characters must match exactly (case included: ``m`` is milli, ``M`` is
    mega); whitespace may differ or be absent on either side. The match may
    not continue a number or an ASCII word on its left (``5V`` is not in
    ``-5V``, ``0.5A``, ``12V``, ``x5V``) and may not be continued by an ASCII
    letter or digit on its right (``12V`` is not in ``12Vin``); Hangul may
    follow (``5V로``). Empty -> ``None``. With ``identifier=True`` (part
    numbers) the match may not stop or start inside a dot / dash-joined code
    either: ``LM2596S-5`` is not in ``LM2596S-5.0/NOPB`` (``LM2596S-5.0`` is:
    a slash separates an option suffix).
    """
    if not quote:
        return None
    chars = [c for c in quote if not c.isspace()]
    if not chars:
        return None
    pattern = re.compile(r"\s*".join(re.escape(c) for c in chars))
    pos = 0
    while pos <= len(raw_input):
        m = pattern.search(raw_input, pos)
        if m is None:
            return None
        s, e = m.start(), m.end()
        if (_boundary_before(raw_input[s - 1] if s > 0 else "") and _boundary_after(raw_input[e] if e < len(raw_input) else "")
                and (not identifier or _identifier_boundary(raw_input, s, e))):
            return s, e
        pos = s + 1
    return None


def quote_in_request(quote: str | None, raw_input: str) -> bool:
    """Whether ``quote`` occurs verbatim in ``raw_input`` at a token boundary (see :func:`find_quote`)."""
    return find_quote(quote, raw_input) is not None


def quote_context(raw_input: str, span: tuple[int, int], width: int = CONTEXT_CHARS) -> str:
    """The request text around ``span`` with the quoted part in brackets: ``…12V 입력을 [5V] 2A로 변환…``."""
    s, e = span
    before = raw_input[max(0, s - width):s]
    after = raw_input[e:e + width]
    return f"{'…' if s - width > 0 else ''}{before}[{raw_input[s:e]}]{after}{'…' if e + width < len(raw_input) else ''}".replace("\n", " ")


_KEY_JUNK = re.compile(r"[^a-z0-9]+")


def canonical_key(key: str) -> str | None:
    """``"Output Voltage"`` -> ``"output_voltage"``; ascii lower-case snake_case, ``None`` when no ascii letter survives."""
    ascii_text = unicodedata.normalize("NFKD", key or "").encode("ascii", "ignore").decode()
    k = _KEY_JUNK.sub("_", ascii_text.lower()).strip("_")
    if not k or not re.search(r"[a-z]", k):
        return None
    return k


def find_directive(*texts: str | None) -> str | None:
    """The first :data:`DIRECTIVE_PHRASES` entry found in any of ``texts``, or ``None``."""
    for t in texts:
        if not t:
            continue
        c = _collapse(t)
        for phrase in DIRECTIVE_PHRASES:
            if phrase in c:
                return phrase
    return None


def request_hash(raw_input: str) -> str:
    """``sha256:<hex>`` of the exact request text - the cache key for an extraction."""
    return "sha256:" + hashlib.sha256(raw_input.encode("utf-8")).hexdigest()


def _sha256(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def extraction_fingerprint() -> dict[str, str]:
    """What a cached model reply depends on besides the request text.

    The extraction version, the prompt *as sent* (system text with the
    schema note, the request framing, the JSON-only instruction a
    non-structured transport adds), the strict schema and the quantity
    parser's rules (they decide what grounds). The model is not part of it:
    the cache is keyed by request text and records ``model`` in the entry.
    """
    prompt = [m.model_dump(mode="json") for m in requirement_extraction_messages("", None)] + [JSON_ONLY_INSTRUCTION]
    return {
        "extraction_version": EXTRACTION_VERSION,
        "prompt_hash": _sha256(json.dumps(prompt, sort_keys=True, ensure_ascii=False)),
        "schema_hash": _sha256(json.dumps(json_schema(), sort_keys=True, ensure_ascii=False)),
        "quantity_version": QUANTITY_VERSION,
    }


def cache_entry_staleness(entry: dict[str, Any] | None) -> str | None:
    """Why a cache entry can not be replayed (``None`` when it is current).

    An entry made under another :data:`EXTRACTION_VERSION`, prompt or schema
    is stale: the model answered a different question. An entry whose stored
    reply no longer validates against :class:`RequirementExtraction` is
    stale too, never a crash.
    """
    if not isinstance(entry, dict):
        return "cache entry is not an object"
    current = extraction_fingerprint()
    for field, want in current.items():
        have = entry.get(field)
        if have != want:
            return f"cached extraction was produced under another {field.replace('_', ' ')} ({have!r} != {want!r}); re-extracting"
    try:
        RequirementExtraction.model_validate(entry.get("extraction"))
    except Exception as e:  # pydantic ValidationError, or a non-dict payload
        first = str(e).splitlines()[0][:200]
        return f"cached extraction no longer matches the schema ({first}); re-extracting"
    if not isinstance(entry.get("model"), str) or not entry["model"].strip():
        return "cached extraction does not record the model that produced it; re-extracting"
    if not isinstance(entry.get("decisions", {}), dict):
        return "cached extraction's decisions are not an object; re-extracting"
    return None


_TRAILING_PUNCT = ".!?。！？,;:~"


def normalise_answer(answer: str | None) -> str:
    """Whitespace collapsed, case folded, trailing punctuation removed: ``" Yes! "`` -> ``"yes"``."""
    if not answer:
        return ""
    return " ".join(answer.split()).casefold().rstrip(_TRAILING_PUNCT).strip()


#: words that add nothing to a confirmation (``ok thanks``, ``네 감사합니다``)
_CONFIRM_FILLER: frozenset[str] = frozenset({"thanks", "thank", "you", "please", "sure", "fine", "good", "감사", "감사합니다", "고마워요", "고맙습니다", "넵", "넹"})


def is_confirmation(answer: str | None) -> bool:
    """Whether a user's answer to :data:`CONFIRM_KEY` confirms.

    The listed spellings (``yes`` / ``y`` / ``ok`` / ``네`` / ``확인`` ...,
    punctuation ignored), or several of them with filler words (``네
    맞습니다``, ``yes, correct``, ``ok thanks``). A word that says anything
    else (``yes, but change X``) is not a confirmation.
    """
    norm = normalise_answer(answer)
    if norm in CONFIRM_ANSWERS:
        return True
    words = [w.strip(_TRAILING_PUNCT + "'\"()") for w in norm.replace(",", " ").split()]
    words = [w for w in words if w]
    return bool(words) and any(w in CONFIRM_ANSWERS for w in words) and all(w in CONFIRM_ANSWERS or w in _CONFIRM_FILLER for w in words)


def is_rejection(answer: str | None) -> bool:
    """Whether the answer rejects the table without saying what is wrong (``no``, ``아니오`` ...)."""
    return normalise_answer(answer) in NEGATIVE_ANSWERS


def is_correction(answer: str | None) -> bool:
    """Whether a non-confirming answer is worth appending to the request as a correction.

    A bare rejection or a very short reply (fewer than
    :data:`MIN_CORRECTION_CHARS` characters) is not: appending ``no`` to the
    request would change its hash and pay for a re-extraction that can not
    improve anything, so the agent asks again instead.
    """
    norm = normalise_answer(answer)
    if not norm or is_confirmation(answer) or is_rejection(answer) or len(norm) < MIN_CORRECTION_CHARS:
        return False
    # ``sure`` / ``fine`` / ``좋습니다``: one word without a digit says yes or no in a spelling the lists lack; it is
    # not a statement about the design, and appending it would change the request's hash and pay for a re-extraction
    if " " not in norm and not any(ch.isdigit() for ch in norm):
        return False
    return True


# --- quantity agreement -------------------------------------------------------


def _agree(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b))


def _model_quantity(number: float | None, unit: str | None, number_high: float | None) -> Quantity | QuantityRange | None:
    """The model's ``number``/``unit`` (``/number_high``) through the deterministic parser; ``None`` when the unit is unknown."""
    if number is None or unit is None or not unit.strip():
        return None
    low = parse_quantity(f"{number!r} {unit.strip()}")
    if not isinstance(low, Quantity):
        return None
    if number_high is None:
        return low
    high = parse_quantity(f"{number_high!r} {unit.strip()}")
    if not isinstance(high, Quantity) or high.unit != low.unit or low.value > high.value:
        return None
    return QuantityRange(low=low.value, high=high.value, unit=low.unit, original=f"{number!r}..{number_high!r} {unit.strip()}")


def _same_quantity(a: Quantity | QuantityRange, b: Quantity | QuantityRange) -> bool:
    if isinstance(a, QuantityRange) or isinstance(b, QuantityRange):
        return isinstance(a, QuantityRange) and isinstance(b, QuantityRange) and a.low == b.low and a.high == b.high and a.unit == b.unit
    return a.value == b.value and a.unit == b.unit and a.plus_minus == b.plus_minus


#: the AC/DC words the quantity parser reads with a volt unit (``12 V DC``); a quote may stop before them
_CURRENT_KIND_WORDS = frozenset({"dc", "ac"})


def _request_quantity(raw_input: str, span: tuple[int, int]) -> tuple[Quantity | QuantityRange | None, str | None]:
    """``(parsed, None)`` or ``(None, reason)`` for the quantity the request states at ``span``.

    The text parsed is the request's own (``raw_input[start:end]``), and the
    result must be exactly what :func:`find_quantities` reads there in the
    whole request - otherwise the quote is a fragment of a larger number or
    range and does not state the user's value.
    """
    start, end = span
    text = raw_input[start:end]
    parsed = parse_quantity(text)
    if parsed is None:
        hits = find_quantities(text)
        if not hits:
            return None, f"no quantity with a unit found in quote {text!r}"
        if len(hits) > 1:
            return None, f"quote {text!r} contains {len(hits)} quantities; expected exactly one"
        return None, f"quote {text!r} contains other numbers besides {format_quantity(hits[0][1])}"
    overlapping = [(s, e, q) for (s, e), q in find_quantities(raw_input) if s < end and e > start]
    if len(overlapping) == 1:
        s, e, q = overlapping[0]
        if start <= s and _same_quantity(q, parsed):
            # ``12V`` quoted from ``12V DC``: the parser reads the AC/DC word with the unit, the quote may leave it out
            if e <= end or raw_input[end:e].strip().strip("()").strip().lower() in _CURRENT_KIND_WORDS:
                return parsed, None
        return None, f"quote {text!r} is part of a larger quantity in the request ({format_quantity(q)})"
    return None, f"quote {text!r} does not read as one quantity in the request ({len(overlapping)} found there)"


def _mismatch(model_q: Quantity | QuantityRange, parsed: Quantity | QuantityRange) -> str | None:
    if isinstance(parsed, Quantity) and parsed.plus_minus:
        # "±5 %": the model may state the magnitude (5 %) or the symmetric range (-5..5 %)
        mag = abs(parsed.value)
        if isinstance(model_q, Quantity):
            if model_q.unit == parsed.unit and _agree(abs(model_q.value), mag):
                return None
        elif model_q.unit == parsed.unit and _agree(model_q.low, -mag) and _agree(model_q.high, mag):
            return None
        return f"quote states a tolerance ({format_quantity(parsed)}) but the model gave {format_quantity(model_q)}"
    if isinstance(parsed, QuantityRange) != isinstance(model_q, QuantityRange):
        what = "a range" if isinstance(parsed, QuantityRange) else "a single value"
        return f"quote states {what} ({format_quantity(parsed)}) but the model gave {format_quantity(model_q)}"
    if model_q.unit != parsed.unit:
        return f"unit mismatch: model {format_quantity(model_q)} vs quote {format_quantity(parsed)}"
    if isinstance(parsed, QuantityRange):
        assert isinstance(model_q, QuantityRange)
        ok = _agree(model_q.low, parsed.low) and _agree(model_q.high, parsed.high)
    else:
        assert isinstance(model_q, Quantity)
        ok = _agree(model_q.value, parsed.value)
    if not ok:
        return f"number mismatch: model {format_quantity(model_q)} vs quote {format_quantity(parsed)}"
    return None


def _traced_payload(q: Quantity | QuantityRange) -> tuple[Any, str]:
    """``(value, unit)`` for a Traced: ranges and ``±`` tolerances become ``[low, high]`` so the marker is data, not prose."""
    if isinstance(q, QuantityRange):
        return [q.low, q.high], q.unit
    if q.plus_minus:
        mag = abs(q.value)
        return [-mag, mag], q.unit
    return q.value, q.unit


def _raw_payload(number: float | None, unit: str | None, number_high: float | None) -> tuple[Any, str | None]:
    """What the model wrote, kept as it wrote it (unit not canonical) - never presented as parsed."""
    if number is None:
        return None, None
    return ([number, number_high] if number_high is not None else number), (unit or None)


def _traced(kind: ProvenanceKind, value: Any, unit: str | None, model: str, note: str) -> Traced:
    return Traced(value=value, unit=unit, provenance=Provenance(kind=kind, tool=model, note=note))


# --- grounding ----------------------------------------------------------------


class GroundedExtraction(BaseModel):
    """What survives grounding, ready to become IR proposals."""

    model: str
    raw_input_hash: str
    requirements: list[Requirement] = Field(default_factory=list)
    questions: list[MissingInformation] = Field(default_factory=list)
    conflicts: list[RequirementConflict] = Field(default_factory=list)
    #: codes from explicit grounded statements only
    jurisdictions: list[str] = Field(default_factory=list)
    application: str | None = None
    #: the verbatim phrase ``application`` was grounded on, as it stands in the request (``None`` when there is no application)
    application_quote: str | None = None
    #: (key, reason) of explicit items that became assumptions
    demoted: list[tuple[str, str]] = Field(default_factory=list)
    #: (key, reason) of items that were not kept at all (also merged duplicates)
    dropped: list[tuple[str, str]] = Field(default_factory=list)
    #: requirement id -> the request text around the quote it was grounded on (grounded explicit items only)
    contexts: dict[str, str] = Field(default_factory=dict)
    #: what else grounding did (items kept apart, ...), for the stage message
    notes: list[str] = Field(default_factory=list)

    @property
    def grounded_explicit(self) -> list[Requirement]:
        return [r for r in self.requirements if is_grounded_explicit(r)]


def is_grounded_explicit(r: Requirement) -> bool:
    """An explicit, given requirement whose value :func:`_ground_explicit` accepted, still awaiting confirmation.

    The provenance note must carry the marker :func:`_ground_explicit` writes
    (``quote: …; parsed: …``): a value that reached an explicit item any other
    way (merged from an implicit item, edited by hand) is not grounded and
    :func:`upgrade_confirmed` leaves it alone.
    """
    if r.kind != RequirementKind.EXPLICIT or r.status != RequirementStatus.GIVEN or r.value is None:
        return False
    p = r.value.provenance
    note = p.note or ""
    return p.kind == ProvenanceKind.LLM_GENERATED and note.startswith(GROUNDED_NOTE_PREFIX) and _GROUNDED_NOTE_MARKER in note


_is_grounded_explicit = is_grounded_explicit


def _grounded_note(quote_text: str, parsed: str, model: str, statement: str | None) -> str:
    note = f"{GROUNDED_NOTE_PREFIX}{quote_text!r}{_GROUNDED_NOTE_MARKER}{parsed}; model: {model}"
    if statement:
        note += f"; model statement: {statement!r}"
    return note


def _ground_explicit(raw_input: str, item: ExtractedRequirement, key: str, model: str) -> tuple[Requirement, str | None, tuple[int, int] | None]:
    """``(requirement, demotion reason or None, span of the value quote in the request or None)``."""
    statement = (item.text or "").strip()
    reason: str | None = None
    parsed: Quantity | QuantityRange | None = None
    span = find_quote(item.quote, raw_input) if (item.quote or "").strip() else None
    value_span: tuple[int, int] | None = None
    if not (item.quote or "").strip():
        reason = "explicit requirement without a quote"
    elif span is None:
        reason = f"quote not found verbatim in request: {item.quote!r}"
    elif item.value is not None:
        v = item.value
        value_span = find_quote(v.quote, raw_input)
        if value_span is None:
            reason = f"value quote not found verbatim in request: {v.quote!r}"
        else:
            parsed, reason = _request_quantity(raw_input, value_span)
            if parsed is not None:
                model_q = _model_quantity(v.number, v.unit, v.number_high)
                if model_q is None:
                    reason = f"model unit not recognised: {v.unit!r} (number {v.number!r})"
                else:
                    reason = _mismatch(model_q, parsed)
    if reason is None:
        assert span is not None
        quote_text = raw_input[span[0]:span[1]]
        if parsed is not None and value_span is not None:
            value, unit = _traced_payload(parsed)
            note = _grounded_note(raw_input[value_span[0]:value_span[1]], format_quantity(parsed), model, statement)
        else:
            value, unit = quote_text, None
            note = _grounded_note(quote_text, "(no numeric value)", model, statement)
            value_span = span
        req = Requirement(
            id=f"req.{key}",
            key=key,
            # the statement is the user's words, never the model's prose (which the note keeps)
            text=f"{key}: {' '.join(quote_text.split())}",
            kind=RequirementKind.EXPLICIT,
            status=RequirementStatus.GIVEN,
            category=item.category,
            value=_traced(ProvenanceKind.LLM_GENERATED, value, unit, model, note),
        )
        return req, None, value_span
    # demote: the number is the model's claim, recorded as an assumption the user must confirm
    text = statement or key
    said = ""
    value: Any
    unit: str | None
    if item.value is not None:
        model_q = _model_quantity(item.value.number, item.value.unit, item.value.number_high)
        if model_q is not None:
            value, unit = _traced_payload(model_q)
            said = f"; model said {format_quantity(model_q)}"
        else:
            value, unit = _raw_payload(item.value.number, item.value.unit, item.value.number_high)
            said = f"; model said {item.value.number!r} {item.value.unit!r} (unit not canonical)"
    else:
        value, unit = text, None
    note = f"demoted from explicit: {reason}{said}; quote: {item.quote!r}; model: {model}"
    req = Requirement(
        id=f"req.{key}",
        key=key,
        text=text,
        kind=RequirementKind.ASSUMPTION,
        status=RequirementStatus.ASSUMED,
        category=item.category,
        value=_traced(ProvenanceKind.ASSUMPTION, value, unit, model, note),
    )
    return req, reason, None


def _ground_implicit(item: ExtractedRequirement, key: str, model: str) -> Requirement:
    text = (item.text or "").strip() or key
    rationale = (item.rationale or "").strip()
    note = f"implicit; rationale: {rationale}"
    value: Any
    unit: str | None
    if item.value is not None:
        model_q = _model_quantity(item.value.number, item.value.unit, item.value.number_high)
        if model_q is not None:
            value, unit = _traced_payload(model_q)
            note += f"; value: {format_quantity(model_q)}"
        else:
            value, unit = _raw_payload(item.value.number, item.value.unit, item.value.number_high)
            note += f"; value: {item.value.number!r} {item.value.unit!r} (unit not canonical, kept as the model wrote it)"
        if item.value.quote:
            note += f"; quote ignored for implicit item: {item.value.quote!r}"
    else:
        value, unit = text, None
    if item.quote:
        note += f"; quote ignored for implicit item: {item.quote!r}"
    note += f"; model: {model}"
    return Requirement(
        id=f"req.{key}",
        key=key,
        text=text,
        kind=RequirementKind.IMPLICIT,
        status=RequirementStatus.ASSUMED,
        category=item.category,
        value=_traced(ProvenanceKind.LLM_GENERATED, value, unit, model, note),
    )


def _ground_assumption(item: ExtractedAssumption, key: str, model: str) -> Requirement:
    text = (item.text or "").strip() or key
    note = f"assumed by model: {item.rationale.strip()}"
    value: Any
    unit: str | None
    model_q = _model_quantity(item.number, item.unit, item.number_high)
    if model_q is not None:
        value, unit = _traced_payload(model_q)
        note += f"; value: {format_quantity(model_q)}"
    elif item.number is not None:
        value, unit = _raw_payload(item.number, item.unit, item.number_high)
        note += f"; value: {item.number!r} {item.unit!r} (unit not canonical, kept as the model wrote it)"
    else:
        value, unit = text, None
    note += f"; model: {model}"
    return Requirement(
        id=f"req.{key}",
        key=key,
        text=text,
        kind=RequirementKind.ASSUMPTION,
        status=RequirementStatus.ASSUMED,
        category=item.category,
        value=_traced(ProvenanceKind.ASSUMPTION, value, unit, model, note),
    )


def _is_numeric(r: Requirement) -> bool:
    if r.value is None:
        return False
    v = r.value.value
    if isinstance(v, bool):
        return False
    return isinstance(v, (int, float)) or (isinstance(v, list) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v))


def _same_value(a: Requirement, b: Requirement) -> bool:
    if _is_numeric(a) and _is_numeric(b):
        assert a.value is not None and b.value is not None
        if (a.value.unit or "") != (b.value.unit or ""):
            return False
        va, vb = a.value.value, b.value.value
        if isinstance(va, list) != isinstance(vb, list):
            return False
        if isinstance(va, list):
            return len(va) == len(vb) and all(_agree(float(x), float(y)) for x, y in zip(va, vb))
        return _agree(float(va), float(vb))
    if a.value is None or b.value is None:
        return a.value is None and b.value is None and _squash(a.text) == _squash(b.text)
    if not _is_numeric(a) and not _is_numeric(b):
        return _squash(str(a.value.value)) == _squash(str(b.value.value))
    return False


def _describe(r: Requirement) -> str:
    if r.value is None:
        return r.text
    v = r.value.value
    if isinstance(v, list):
        body = "..".join(f"{float(x):.12g}" for x in v)
    elif isinstance(v, (int, float)) and not isinstance(v, bool):
        body = f"{float(v):.12g}"
    else:
        body = repr(v)
    return f"{body} {r.value.unit}" if r.value.unit else body


def _same_kind(a: Requirement, b: Requirement) -> bool:
    """Same kind, status and provenance kind: the only items whose values may be merged."""
    pa = a.value.provenance.kind if a.value is not None else None
    pb = b.value.provenance.kind if b.value is not None else None
    return a.kind == b.kind and a.status == b.status and pa == pb


def _rank(r: Requirement) -> int:
    """Which duplicate holds the plain ``req.<key>`` id: grounded explicit first, then implicit, then assumptions."""
    if _is_grounded_explicit(r):
        return 0
    if r.kind == RequirementKind.IMPLICIT:
        return 1
    return 2


def _join_text(first: Requirement, other: Requirement) -> str:
    if _squash(other.text) in _squash(first.text):
        return first.text
    prefix = f"{first.key}: "
    if first.text.startswith(prefix) and other.text.startswith(prefix):
        return f"{first.text}; {other.text[len(prefix):]}"
    return f"{first.text}; {other.text}"


def _dedupe(candidates: list[Requirement], dropped: list[tuple[str, str]], notes: list[str]) -> tuple[list[Requirement], list[RequirementConflict]]:
    """Merge duplicate keys (identical values, or same-kind items with at most one value); conflict or keep apart otherwise."""
    out: list[Requirement] = []
    first_of: dict[str, Requirement] = {}
    conflicts: list[RequirementConflict] = []
    for req in candidates:
        first = first_of.get(req.key)
        if first is None:
            first_of[req.key] = req
            out.append(req)
            continue
        if _same_value(first, req):
            if _rank(req) < _rank(first):
                # the grounded explicit statement survives whatever order the model listed the two in: it is the
                # one the user can confirm into user_requirement, the inference is not
                req.id = first.id
                out[out.index(first)] = req
                first_of[req.key] = req
                dropped.append((first.key, f"duplicate of {req.id} with an identical value; merged into the {req.kind} item"))
            else:
                dropped.append((req.key, f"duplicate of {first.id} with an identical value; merged"))
            continue
        if not (_is_numeric(first) and _is_numeric(req)):
            if _same_kind(first, req):
                # at most one numeric value between two items of one kind: keep it, join the statements
                if _is_numeric(req) and not _is_numeric(first):
                    first.value = req.value
                first.text = _join_text(first, req)
                dropped.append((req.key, f"duplicate of {first.id} without a competing value; merged"))
                continue
            # different kinds: a value never moves between an explicit statement and a model inference or
            # assumption (it would ride the explicit item into user_requirement on confirmation). Keep both;
            # the higher-ranked item holds the plain id.
            taken = {r.id for r in out}
            if _rank(req) < _rank(first):
                first.id = _unique_id(f"req.{first.key}.{first.kind}", taken - {first.id})
                req.id = f"req.{req.key}"
                first_of[req.key] = req
                apart, primary = first, req
            else:
                req.id = _unique_id(f"req.{req.key}.{req.kind}", taken)
                apart, primary = req, first
            out.append(req)
            notes.append(f"{req.key}: {apart.kind} item kept apart as {apart.id} from the {primary.kind} one (values never move between kinds)")
            continue
        n = sum(1 for r in out if r.key == req.key) + 1
        alt = req.model_copy(update={"id": _unique_id(f"req.{req.key}.alt{n}", {r.id for r in out}), "status": RequirementStatus.CONFLICTING})
        first.status = RequirementStatus.CONFLICTING
        out.append(alt)
        conflicts.append(
            RequirementConflict(
                requirement_ids=[first.id, alt.id],
                description=f"{req.key}: {_describe(first)} ({first.kind}) vs {_describe(alt)} ({alt.kind})",
            )
        )
    return out, conflicts


def _unique_id(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    n = 2
    while f"{base}{n}" in taken:
        n += 1
    return f"{base}{n}"


_JURISDICTION_CODE = re.compile(r"^[A-Z]{2,3}$")


def jurisdiction_named_in(code: str, text: str) -> bool:
    """Whether ``text`` names the jurisdiction ``code``: the code as an upper-case token, or a :data:`JURISDICTION_NAMES` entry."""
    if re.search(rf"(?<![A-Za-z0-9-]){re.escape(code)}(?![A-Za-z0-9-])", text):
        return True
    folded = _collapse(text)
    for name in JURISDICTION_NAMES.get(code, ()):
        if name.isascii():
            if re.search(rf"(?<![a-z]){re.escape(name)}(?![a-z])", folded):
                return True
        elif name in folded:
            return True
    return False


def _cap_question(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_QUESTION_CHARS:
        return text
    return text[:MAX_QUESTION_CHARS].rstrip() + " … [model question cut at " + str(MAX_QUESTION_CHARS) + " characters]"


def ground_extraction(raw_input: str, extraction: RequirementExtraction, model: str) -> GroundedExtraction:
    """Check every claim in ``extraction`` against ``raw_input`` (see the module docstring for the rules)."""
    demoted: list[tuple[str, str]] = []
    dropped: list[tuple[str, str]] = []
    notes: list[str] = []
    candidates: list[Requirement] = []
    spans: dict[int, tuple[int, int]] = {}  # id(requirement) -> span of its value quote

    for item in extraction.requirements:
        key = canonical_key(item.key)
        if key is None:
            dropped.append((item.key, "key has no ascii letters after canonicalisation"))
            continue
        hit = find_directive(item.key, item.text, item.quote, item.rationale, item.value.quote if item.value else None, item.value.unit if item.value else None)
        if hit is not None:
            dropped.append((key, f"directive phrase {hit!r} in extracted text; the request is data, not instructions"))
            continue
        if item.kind == "explicit":
            req, reason, span = _ground_explicit(raw_input, item, key, model)
            if reason is not None:
                demoted.append((key, reason))
            elif span is not None:
                spans[id(req)] = span
            candidates.append(req)
        else:
            if not (item.rationale or "").strip():
                dropped.append((key, "implicit requirement without a rationale"))
                continue
            candidates.append(_ground_implicit(item, key, model))

    for a in extraction.assumptions:
        key = canonical_key(a.key)
        if key is None:
            dropped.append((a.key, "key has no ascii letters after canonicalisation"))
            continue
        hit = find_directive(a.key, a.text, a.rationale, a.unit)
        if hit is not None:
            dropped.append((key, f"directive phrase {hit!r} in extracted text; the request is data, not instructions"))
            continue
        if not (a.rationale or "").strip():
            dropped.append((key, "assumption without a rationale"))
            continue
        candidates.append(_ground_assumption(a, key, model))

    requirements, conflicts = _dedupe(candidates, dropped, notes)
    contexts = {r.id: quote_context(raw_input, spans[id(r)]) for r in requirements if id(r) in spans}
    ids = {r.id for r in requirements}
    keys = {r.key for r in requirements}

    for c in extraction.conflicts:
        hit = find_directive(c.description, *c.keys)
        if hit is not None:
            dropped.append(("conflict", f"directive phrase {hit!r} in extracted text; the request is data, not instructions"))
            continue
        ckeys = [canonical_key(k) for k in c.keys]
        if not ckeys or any(k is None or k not in keys for k in ckeys):
            dropped.append(("conflict", f"conflict references unknown keys {c.keys!r}: {c.description}"))
            continue
        ref_ids = [f"req.{k}" for k in ckeys if f"req.{k}" in ids]
        conflicts.append(RequirementConflict(requirement_ids=ref_ids, description=c.description.strip()))

    grounded_keys = {r.key for r in requirements if _is_grounded_explicit(r)}
    questions: list[MissingInformation] = []
    seen: set[str] = set()
    for q in extraction.questions:
        key = canonical_key(q.key)
        if key is None:
            dropped.append((q.key, "question key has no ascii letters after canonicalisation"))
            continue
        hit = find_directive(q.key, q.question, q.rationale, *q.options)
        if hit is not None:
            dropped.append((key, f"directive phrase {hit!r} in extracted text; the request is data, not instructions"))
            continue
        if key == CONFIRM_KEY:
            dropped.append((key, "reserved question key"))
            continue
        if key in grounded_keys:
            dropped.append((key, "question already answered by a grounded explicit requirement"))
            continue
        if key in seen:
            dropped.append((key, "duplicate question key; first kept"))
            continue
        if not q.question.strip():
            dropped.append((key, "empty question"))
            continue
        seen.add(key)
        questions.append(
            MissingInformation(
                key=key,
                question=_cap_question(q.question),
                required=q.required,
                options=[o.strip()[:MAX_QUESTION_CHARS] for o in q.options if o.strip()],
                rationale=(q.rationale or "").strip()[:MAX_QUESTION_CHARS],
                source="llm",
            )
        )

    jurisdictions: list[str] = []
    for j in extraction.jurisdictions:
        code = (j.code or "").strip().upper()
        hit = find_directive(j.code, j.quote)
        if hit is not None:
            dropped.append((f"jurisdiction:{code}", f"directive phrase {hit!r} in extracted text; the request is data, not instructions"))
            continue
        if not _JURISDICTION_CODE.match(code):
            dropped.append((f"jurisdiction:{j.code}", "jurisdiction code must be 2-3 upper-case letters (EU, US, KR)"))
            continue
        jspan = find_quote(j.quote, raw_input)
        if jspan is None:
            dropped.append((f"jurisdiction:{code}", f"quote not found verbatim in request: {j.quote!r}"))
            continue
        quote_text = raw_input[jspan[0]:jspan[1]]
        if not jurisdiction_named_in(code, quote_text):
            dropped.append((f"jurisdiction:{code}", f"quote {quote_text!r} does not name {code}"))
            continue
        if code not in jurisdictions:
            jurisdictions.append(code)

    application: str | None = None
    application_quote: str | None = None
    if extraction.application is not None:
        app = extraction.application
        hit = find_directive(app.summary, app.quote)
        aspan = find_quote(app.quote, raw_input)
        if hit is not None:
            dropped.append(("application", f"directive phrase {hit!r} in extracted text; the request is data, not instructions"))
        elif aspan is None:
            dropped.append(("application", f"quote not found verbatim in request: {app.quote!r}"))
        elif not app.summary.strip():
            dropped.append(("application", "empty application summary"))
        else:
            application = app.summary.strip()
            application_quote = raw_input[aspan[0]:aspan[1]]
            contexts["req.application"] = quote_context(raw_input, aspan)

    return GroundedExtraction(
        model=model,
        raw_input_hash=request_hash(raw_input),
        requirements=requirements,
        questions=questions,
        conflicts=conflicts,
        jurisdictions=jurisdictions,
        application=application,
        application_quote=application_quote,
        demoted=demoted,
        dropped=dropped,
        contexts=contexts,
        notes=notes,
    )


def application_requirement(grounded: GroundedExtraction) -> Requirement | None:
    """``req.application`` for a grounded application statement (``llm_generated`` until confirmed), else ``None``.

    The value is the model's summary; the note keeps the verbatim quote it was
    grounded on, so :func:`upgrade_confirmed` treats it like any other grounded
    explicit item and the baseline ``application`` question is answered.
    """
    if grounded.application is None:
        return None
    note = _grounded_note(grounded.application_quote or "", "(application statement)", grounded.model, None)
    return Requirement(
        id="req.application",
        key="application",
        text=f"application: {grounded.application}",
        kind=RequirementKind.EXPLICIT,
        status=RequirementStatus.GIVEN,
        category="application",
        value=_traced(ProvenanceKind.LLM_GENERATED, grounded.application, None, grounded.model, note),
    )


# --- confirmation -------------------------------------------------------------


def upgrade_confirmed(requirements: list[Requirement]) -> list[Requirement]:
    """Grounded explicit values re-tagged ``user_requirement`` (note keeps model + quote); everything else unchanged.

    Implicit requirements stay ``llm_generated`` and assumptions stay
    ``assumption``: the user confirmed what they *said*, not what the model
    inferred or assumed. Those are decided one by one with
    :func:`decide_inferred`.
    """
    out: list[Requirement] = []
    for r in requirements:
        if not _is_grounded_explicit(r):
            out.append(r)
            continue
        assert r.value is not None
        old = r.value.provenance
        note = old.note or ""
        if old.tool and f"model: {old.tool}" not in note:
            note = f"{note}; model: {old.tool}" if note else f"model: {old.tool}"
        new_prov = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note=f"{CONFIRMED_NOTE_PREFIX}; {note}")
        out.append(r.model_copy(update={"value": r.value.model_copy(update={"provenance": new_prov})}))
    return out


#: how :func:`decide_inferred` marks an accepted implicit / assumed item's note
ACCEPTED_NOTE_PREFIX = "accepted by user"


def is_inferred(r: Requirement) -> bool:
    """A requirement the model inferred or assumed (implicit, or an assumption incl. demoted explicit claims), still undecided."""
    if r.value is None:
        return False
    return r.kind in (RequirementKind.IMPLICIT, RequirementKind.ASSUMPTION) and r.value.provenance.needs_verification and bool(r.value.provenance.tool)


def decide_inferred(requirements: list[Requirement], accept: set[str], reject: set[str]) -> tuple[list[Requirement], list[str]]:
    """Apply the user's per-item decisions on inferred items: accepted ones become ``user_requirement``, rejected ones are removed.

    Returns ``(requirements, notes)``. A key that names no inferred item is
    reported, never guessed at; a key in both sets is rejected (the safer
    reading) and reported.
    """
    notes: list[str] = []
    both = accept & reject
    for k in sorted(both):
        notes.append(f"{k}: named in both accept and reject; treated as rejected")
    accept = accept - both
    out: list[Requirement] = []
    seen: set[str] = set()
    for r in requirements:
        if is_inferred(r) and r.key in reject:
            seen.add(r.key)
            notes.append(f"{r.key}: {r.kind} item rejected by the user and removed")
            continue
        if is_inferred(r) and r.key in accept:
            seen.add(r.key)
            assert r.value is not None
            old = r.value.provenance
            note = old.note or ""
            if old.tool and f"model: {old.tool}" not in note:
                note = f"{note}; model: {old.tool}" if note else f"model: {old.tool}"
            new_prov = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note=f"{ACCEPTED_NOTE_PREFIX}; {note}")
            out.append(r.model_copy(update={"status": RequirementStatus.GIVEN, "value": r.value.model_copy(update={"provenance": new_prov})}))
            notes.append(f"{r.key}: {r.kind} item accepted by the user as a requirement")
            continue
        out.append(r)
    for k in sorted((accept | reject) - seen):
        notes.append(f"{k}: no undecided implicit / assumed requirement with that key; ignored")
    return out, notes


def from_extraction(r: Requirement) -> bool:
    """Whether a requirement in the IR came from an extraction (grounded, implicit, demoted/assumed, confirmed or accepted).

    Model-tagged ``llm_generated`` / ``assumption`` values (``provenance.tool``
    is the model) and ``user_requirement`` values carrying the
    :data:`CONFIRMED_NOTE_PREFIX` or :data:`ACCEPTED_NOTE_PREFIX` note. A
    typed user answer (``tool`` unset, no such note) is not.
    """
    if r.value is None:
        return False
    p = r.value.provenance
    if p.kind in (ProvenanceKind.LLM_GENERATED, ProvenanceKind.ASSUMPTION):
        return bool(p.tool)
    if p.kind == ProvenanceKind.USER_REQUIREMENT:
        return (p.note or "").startswith((CONFIRMED_NOTE_PREFIX, ACCEPTED_NOTE_PREFIX))
    return False


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip()
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(c.ljust(widths[i]) for i, c in enumerate(r)).rstrip() for r in rows]
    return "\n".join([line, sep, *body])


def confirmation_question(grounded: GroundedExtraction) -> MissingInformation:
    """The required ``confirm_requirements`` question: a table of every extracted item for the user to confirm or correct.

    Every grounded explicit item is shown with its quote *in the request's
    own context* so a wrong key assignment (``12V`` filed under
    ``output_voltage``) is visible; the basis column carries the provenance
    note (quote, deterministic parse, model, the model's own statement).
    """
    rows: list[list[str]] = []
    for r in grounded.requirements:
        prov = r.value.provenance if r.value is not None else None
        basis = prov.note or "" if prov is not None else ""
        rows.append([r.key, f"{r.kind}/{r.status}", _describe(r), r.category, grounded.contexts.get(r.id, "-"), basis])
    parts = [
        f"Requirements extracted by {grounded.model} from your request. Nothing below is trusted yet: "
        "explicit items become your requirements only when you confirm them; assumptions and implicit items stay "
        f"unverified (decide them one by one with {ACCEPT_KEY} / {REJECT_KEY}). The key and category of each "
        "row are the model's assignment - check them against the 'in your request' column.",
        "",
        _table(["key", "kind/status", "value", "category", "in your request", "basis"], rows) if rows else "(no requirements extracted)",
    ]
    if grounded.conflicts:
        parts += ["", "Conflicts:", *[f"  - {c.description} [{', '.join(c.requirement_ids)}]" for c in grounded.conflicts]]
    app_ctx = grounded.contexts.get("req.application")
    parts += ["", f"Application: {grounded.application or '(not stated)'}" + (f"  (in your request: {app_ctx})" if app_ctx else "")]
    parts += [f"Jurisdictions: {', '.join(grounded.jurisdictions) or '(not stated)'}"]
    if grounded.demoted:
        parts += ["", "Demoted to assumptions (could not be grounded in your words):", *[f"  - {k}: {why}" for k, why in grounded.demoted]]
    if grounded.dropped:
        parts += ["", "Dropped:", *[f"  - {k}: {why}" for k, why in grounded.dropped]]
    if grounded.notes:
        parts += ["", "Notes:", *[f"  - {n}" for n in grounded.notes]]
    parts += [
        "",
        "Reply yes / y / ok / confirm (네 / 예 / 확인) to confirm the explicit items as your requirements, "
        "or describe what is wrong; a correction is appended to your request and the extraction is run again.",
    ]
    return MissingInformation(
        key=CONFIRM_KEY,
        question="\n".join(parts),
        required=True,
        rationale="LLM extractions are proposals; only the user can make them requirements",
    )


# --- prompt -------------------------------------------------------------------

#: the system prompt text lives in :mod:`ai_eda.llm.prompts`; kept under its old name for callers
EXTRACTION_SYSTEM_PROMPT = REQUIREMENT_EXTRACTION_SYSTEM


def build_extraction_messages(raw_input: str, answers: dict[str, str] | None = None, *, include_schema: bool = True) -> list[LLMMessage]:
    """System + user messages for the extraction call; the request and the known answers are framed as data.

    ``include_schema=False`` when the service enforces the schema through
    ``response_format`` (the prompt then says so instead of embedding it).
    """
    return requirement_extraction_messages(raw_input, answers, schema=json_schema() if include_schema else None)


__all__ = [
    "ACCEPTED_NOTE_PREFIX",
    "ACCEPT_KEY",
    "REJECT_KEY",
    "CONFIRM_ANSWERS",
    "CONFIRM_KEY",
    "CONFIRMED_NOTE_PREFIX",
    "CONTEXT_CHARS",
    "DIRECTIVE_PHRASES",
    "EXTRACTION_SYSTEM_PROMPT",
    "EXTRACTION_VERSION",
    "GROUNDED_NOTE_PREFIX",
    "JURISDICTION_NAMES",
    "MAX_QUESTION_CHARS",
    "MIN_CORRECTION_CHARS",
    "NEGATIVE_ANSWERS",
    "REL_TOL",
    "Category",
    "ExtractedApplication",
    "ExtractedAssumption",
    "ExtractedConflict",
    "ExtractedJurisdiction",
    "ExtractedQuestion",
    "ExtractedRequirement",
    "ExtractedValue",
    "GroundedExtraction",
    "RequirementExtraction",
    "application_requirement",
    "build_extraction_messages",
    "cache_entry_staleness",
    "canonical_key",
    "confirmation_question",
    "decide_inferred",
    "extraction_fingerprint",
    "find_directive",
    "find_quote",
    "from_extraction",
    "ground_extraction",
    "is_confirmation",
    "is_correction",
    "is_grounded_explicit",
    "is_inferred",
    "is_rejection",
    "json_schema",
    "jurisdiction_named_in",
    "normalise_answer",
    "quote_context",
    "quote_in_request",
    "request_hash",
    "upgrade_confirmed",
]
