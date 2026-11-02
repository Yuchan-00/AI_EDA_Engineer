"""Shared types of the deterministic circuit templates: the plan a template produces and how its values are stamped.

Invariant: a template's free choices (a resistor it picked, a tolerance, a
sweep grid, a modelling decision) are **not derived from anything** and are
never stamped ``derived``. They are shown to the user as a table under the
required question ``confirm_design`` and become the user's values
(``user_requirement``, note :data:`CHOICE_NOTE_PREFIX` + template id and
version) only when that table is confirmed; until then they carry
``assumption`` provenance and the plan is not applied. Only numbers a
registered calculator computed from traced inputs are ``derived`` (with the
calculator's tool id and role map, so ``calc.recompute`` re-derives them).
Structural decisions (which parts, which nets, which analysis) are ``derived``
by the template tool itself (``design.template.<id>``): a pure function of
the template id the user confirmed and of the library on disk.

A template also *explains* its design for the stage reports
(:mod:`ai_eda.report.stages`): :meth:`Template.theory` gives the circuit
theory with the IR's own numbers substituted, :meth:`Template.theory_figures`
draws the curves that theory text derives (a
:class:`~ai_eda.report.figures.Figure` each, captioned with the equation the
text names) and :meth:`Template.part_notes` the role, the reason and the
substitute criteria of every part. All three are views: they read
``ir.parameters`` through :func:`parameter_value` (a missing key prints
:data:`NO_RECORD` or draws no figure, never a guess), recompute display
numbers with the formulas they show and write nothing into the IR.
:meth:`Template.theory_figure_vectors` only *names* the SPICE vectors the
theory report should plot beside the expectations' own (the template's
base nets, say); the report reads them from the recorded run, never from the
template. Substitute part names are suggestions the pipeline never verified
and carry :data:`UNVERIFIED_SUBSTITUTE` on every line.

Leaving requirements out. The world of a template is closed: a confirmed
design requirement it does not serve refuses it. Every such refusal names the
exact answer that leaves the requirement out (:func:`leave_out_remedy`:
``--answer leave_out=<key>``, one key or all refused keys at once) - never
only "edit the IR". The requirement agent moves a left-out requirement into
``ir.requirements.left_out`` (the user's recorded decision), so no reader
here sees it; the confirmation table lists it (:attr:`Plan.left_out`, part of
the table's hash) and a later typed answer to its key brings it back.

Selection hooks. :meth:`Template.triggered_by` decides whether the confirmed
requirements select a template; the default is :meth:`Template.triggered` on
the numeric inputs, so every template that does not override it is selected
exactly as before. A template selected by a categorical requirement (the
radio family, by ``radio_build``) overrides it and reads that requirement
itself. :attr:`Template.layer_policy` (:class:`LayerPolicy`) names the board
layer counts a template builds and the one it uses when no requirement
states a count; the default is every generic stack
(:data:`~ai_eda.design.inputs.LAYER_COUNT_OPTIONS`) with
:data:`~ai_eda.design.inputs.DEFAULT_LAYER_COUNT`, so a template without its
own policy builds exactly the stack it built before. A policy that allows
fewer counts says why, and a stated count outside it refuses the template
with that reason before any missing input is asked for (a required question
whose only outcome is a refusal is never asked).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ai_eda.ir import CircuitIR, Keepout, MissingInformation, Provenance, ProvenanceKind, Requirement, Traced
from ai_eda.tools.kicad.library import KicadLibrary

from ai_eda.design.inputs import (
    BOARD_KEY_ALIASES,
    CATEGORICAL_KEYS,
    DEFAULT_LAYER_COUNT,
    KEY_ALIASES,
    LAYER_COUNT_OPTIONS,
    UNIT_OF,
    DesignInput,
    canonical_key,
    read_inputs,
    read_value,
)

if TYPE_CHECKING:  # the report package imports this one (stages -> TEMPLATES), so the figure type is a type-only import here
    from ai_eda.design.board import BoardContext, SIDeclarations
    from ai_eda.report.figures import Figure

#: 0.2: part values are KiCad-style display text to 5 significant digits (``100n``, not the netlist spelling ``1e-7``);
#: 0.3: every template also proposes the board stack (``pcb_layers`` 2 | 4) and its signal-integrity net classes (``ir.si``)
TEMPLATE_VERSION = "0.3"
#: tool id of the template machinery (``Provenance.tool`` is ``design.template.<template id>`` on structural decisions)
TOOL_ID = "design.template"
#: how a confirmed free choice's note starts (the rest names the template, its version and the choice)
CHOICE_NOTE_PREFIX = "design choice confirmed by user"
#: the answer key that confirms a presented plan; a control key (never a requirement), see ``ai_eda.agents.requirement.CONTROL_KEYS``
CONFIRM_DESIGN_KEY = "confirm_design"
#: the answer key that leaves named requirements out of the design (``--answer leave_out=<key>,<key>``); a control key (never a
#: requirement): the requirement agent moves each named requirement into ``ir.requirements.left_out`` (the user's recorded
#: decision), so no template reads it and no closed world refuses on it. Every closed-world refusal names this answer
#: (:func:`leave_out_answer`). Defined here because the templates name it and must not import the agents.
LEAVE_OUT_KEY = "leave_out"
#: what a report prints for a number the IR does not hold (a report never guesses one)
NO_RECORD = "기록 없음"
#: the marker every substitute-part line carries: the pipeline verified nothing about that part
UNVERIFIED_SUBSTITUTE = "(검증되지 않음: 핀 배열·정격을 데이터시트와 KiCad 라이브러리에서 확인)"
#: IR unit spellings shown with their symbol in a report
UNIT_DISPLAY: dict[str, str] = {"ohm": "Ω", "degC": "°C", "deg": "°", "percent": "%"}
_SI_PREFIXES: dict[int, str] = {-12: "p", -9: "n", -6: "µ", -3: "m", 0: "", 3: "k", 6: "M", 9: "G"}

#: requirement categories a template must serve or refuse (the reviewer's ``requirements_vs_ir`` design categories)
DESIGN_CATEGORIES: frozenset[str] = frozenset({"electrical", "thermal", "mechanical", "signal_integrity", "power_integrity", "rf"})
#: requirement keys every template may ignore: they describe the product, not the circuit
IGNORED_KEYS: frozenset[str] = frozenset({"application", "jurisdiction"})
#: canonical board keys every template serves through the board stackup (:mod:`ai_eda.design.stackup`): the layer
#: count ``pcb_layers`` (read by :func:`~ai_eda.design.inputs.read_layer_count`) is a board decision, not a circuit one
BOARD_KEYS: frozenset[str] = frozenset(BOARD_KEY_ALIASES)
#: the keys a template's ``needs`` may name: what :func:`~ai_eda.design.inputs.present_keys` can count as present
_NEEDABLE_KEYS: frozenset[str] = frozenset(KEY_ALIASES) | frozenset(CATEGORICAL_KEYS)
#: bare requirement keys that name a physical quantity but not *which* one (bare key -> its unit): never an alias of a
#: template input, so no template reads them; the circuit agent asks which specific key is meant (:func:`specific_keys`)
#: and a template counts one as served only beside the specific key stated with the same number
#: (:func:`served_through_specific_key`). ``frequency`` could be an oscillator's output, a filter corner, an MCU clock
#: or an RF carrier - reading it as any one of them would be a guess
AMBIGUOUS_KEYS: dict[str, str] = {"frequency": "Hz"}


def template_tool(template_id: str) -> str:
    return f"{TOOL_ID}.{template_id}"


def leave_out_answer(keys: list[str] | tuple[str, ...]) -> str:
    """The exact answer that leaves the requirements under ``keys`` out of the design: ``--answer leave_out=k1,k2`` (order kept, repeats dropped)."""
    return f"--answer {LEAVE_OUT_KEY}={','.join(dict.fromkeys(keys))}"


def leave_out_remedy(key: str, keys: list[str] | tuple[str, ...] = (), where: str = "this design") -> str:
    """The closed-world sentence that names the exact answer leaving ``key`` out (and all refused ``keys`` at once, when more than one).

    A refused requirement is always leavable by an answer: the leave-out is
    recorded as the user's decision (``ir.requirements.left_out``), the
    template then builds without it, and a later typed answer to the same key
    brings it back.
    """
    together = list(dict.fromkeys(keys))
    both = f" (every requirement refused here at once: {leave_out_answer(together)})" if len(together) > 1 else ""
    return (f"Leave it out of {where} with {leave_out_answer([key])}{both} - recorded in the IR as your decision, the template then builds "
            f"without it and a later typed answer to {key} brings it back")


def template_reads_key(key: str) -> bool:
    """Whether some template can read a requirement under ``key``: a canonical key or one of its aliases (quantity, board or categorical key).

    Nothing is guessed from the spelling: ``battery_voltage_max`` or
    ``antenna_switch_present`` read like design keys but no template reads them.
    """
    return canonical_key(key) is not None


def left_out_lines(ir: CircuitIR) -> list[str]:
    """One line per requirement the user left out of the design (``ir.requirements.left_out``), for the confirmation table and the notes."""
    out: list[str] = []
    for x in ir.requirements.left_out:
        if x.requirement is not None:
            # a value that never became the user's says whose it is (a model's text is labelled wherever it is shown)
            p = x.requirement.value.provenance if x.requirement.value is not None else None
            whose = f" ({p.kind.value}: never confirmed by you)" if p is not None and p.needs_verification else ""
            out.append(f"{x.requirement.id}: {requirement_text(x.requirement)} [{x.requirement.category}]{whose}")
        elif x.question is not None:
            whose = "the model's open question" if x.question.source == "llm" else "the open question"
            out.append(f"{x.key}: (no requirement) {whose} '{x.question.question[:120]}' closed without an answer")
        else:
            out.append(f"{x.key}: (no requirement)")
    return out


def refusal_question(key: str, question: str, rationale: str) -> MissingInformation:
    """A closed-world refusal as a question: non-required, under the requirement's own ``key``, answered by ``--answer leave_out=<key>``.

    ``answer_key`` marks it (:attr:`~ai_eda.ir.MissingInformation.answer_key`),
    so ``ai-eda report`` offers ``--answer leave_out=<key>`` and the GUI a
    leave-out control - never a value under the refused key, which is the
    dead end the refusal is about (a typed value is kept, or replaced and
    refused again).
    """
    return MissingInformation(key=key, required=False, question=question, rationale=rationale, answer_key=LEAVE_OUT_KEY)


def design_view_outside_requirements(ir: CircuitIR) -> dict[str, Any]:
    """The design view without ``requirements``: what :func:`design_references` searches (build it once for many ids)."""
    data = ir.design_dict()
    data.pop("requirements", None)
    return data


def design_references(ir: CircuitIR, requirement_id: str, limit: int = 3, *, data: dict[str, Any] | None = None) -> list[str]:
    """Where the design view outside ``requirements`` holds ``requirement_id`` as a value (``components[0].serves_requirements[0]``, ...).

    Every place a design element names a requirement - ``serves_requirements``,
    a parameter's ``derived_from``, an expectation's ``requirement_id``, a
    stimulus, an RF row - holds the id as one whole string, so an exact match
    over the design view finds them all without naming the fields. ``data``
    is that view when the caller already built it (:func:`design_view_outside_requirements`,
    one build per run). A requirement the design references is never left
    out (it would untrace the design), so no message offers that answer for it.
    """
    if data is None:
        data = design_view_outside_requirements(ir)
    found: list[str] = []

    def walk(node: Any, path: str) -> None:
        if len(found) >= limit:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif node == requirement_id:
            found.append(path)

    walk(data, "")
    return found


def leavable_keys(ir: CircuitIR, requirements: list[Requirement]) -> list[str]:
    """The recorded keys of ``requirements`` a ``leave_out`` answer can take: design-category, not referenced by the design (order kept)."""
    view: dict[str, Any] | None = None
    keys: list[str] = []
    for r in requirements:
        if r.category not in DESIGN_CATEGORIES or r.key in keys:
            continue
        if view is None:
            view = design_view_outside_requirements(ir)
        if not design_references(ir, r.id, limit=1, data=view):
            keys.append(r.key)
    return keys


def unusable_remedy(ir: CircuitIR, aliases: frozenset[str] | set[str] | tuple[str, ...] | list[str]) -> str:
    """The answers that clear a stated key no reader can use (ambiguous, unreadable): leave out one of the requirements stating it.

    ``aliases`` are the keys the reader reads for it (``KEY_ALIASES[canon]``,
    the layer-count / modulation / radio-build aliases). A value the user
    typed earlier is kept, so re-typing the key never clears it; a leave-out
    does. Each distinct recorded key is its own alternative - ``leave_out``
    matches a requirement's key exactly, so the canonical key never stands in
    for an alias - and names the requirements it keeps. When one key holds
    them all, leaving it out takes them all and the key is answered again in
    a later run. ``""`` when no requirement states it.
    """
    reqs = [r for r in ir.requirements.requirements if r.key in aliases]
    keys = list(dict.fromkeys(r.key for r in reqs))
    if not keys:
        return ""
    if len(keys) == 1:
        return f"a typed value is kept, so change it by leaving it out with {leave_out_answer(keys)} and answering {keys[0]} again in a later run"
    alts = " or ".join(f"{leave_out_answer([k])} (keeps {', '.join(r.id for r in reqs if r.key != k)})" for k in keys)
    return f"leave one of them out: {alts}"


def with_remedy(text: str | None, remedy: str) -> str | None:
    """``text; remedy`` (``text`` alone without a remedy, ``None`` without a text)."""
    if text is None:
        return None
    return f"{text}; {remedy}" if remedy else text


def unusable_note(ir: CircuitIR | None, key: str, why: str) -> str:
    """``<key> not usable: <why>``, and - with ``ir`` - the answer that clears it (:func:`~ai_eda.design.base.unusable_remedy`).

    The reason text itself is left as :func:`~ai_eda.design.inputs.read_inputs`
    wrote it (other checks quote it); the remedy is appended to the note: a
    value typed earlier is kept, so only a leave-out of one of the
    requirements stating the key clears it.
    """
    remedy = unusable_remedy(ir, KEY_ALIASES.get(key, (key,))) if ir is not None else ""
    return f"{key} not usable: {why}" + (f"; {remedy}" if remedy else "")


def alias_duplicates(ir: CircuitIR, inputs: dict[str, DesignInput]) -> dict[str, list[Requirement]]:
    """Per numeric input read, the *other* requirements under its aliases (``req.battery_voltage`` beside ``req.input_voltage``).

    :func:`~ai_eda.design.inputs.read_inputs` reads every one of them and
    uses the key only when all read the same number, so each is served by the
    same parts as the requirement the input names: the circuit agent traces
    it to them and the table lists it - a template's closed world counts it as
    served, so the reviewer must see it traced too.
    """
    out: dict[str, list[Requirement]] = {}
    for canon, inp in inputs.items():
        aliases = KEY_ALIASES.get(canon)
        if aliases is None:
            continue
        others = [r for r in ir.requirements.requirements if r.key in aliases and r.id != inp.requirement.id]
        if others:
            out[canon] = others
    return out


def structural_provenance(template_id: str, note: str) -> Provenance:
    """Provenance of a structural decision the template made (a part, a net, the topology, an analysis)."""
    return Provenance(kind=ProvenanceKind.DERIVED, tool=template_tool(template_id), tool_version=TEMPLATE_VERSION, note=note)


def choice_provenance(template_id: str, description: str, confirmed: bool) -> Provenance:
    """Provenance of a free choice: the user's once the table was confirmed, an assumption (never applied) before."""
    if confirmed:
        return Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note=f"{CHOICE_NOTE_PREFIX}; template {template_id} v{TEMPLATE_VERSION}: {description}")
    return Provenance(kind=ProvenanceKind.ASSUMPTION, note=f"template {template_id} v{TEMPLATE_VERSION} choice, not yet confirmed: {description}")


@dataclass(frozen=True)
class Choice:
    """A free design choice of a template, shown in the confirmation table.

    ``value`` / ``unit`` are the number (or string) the choice fixes; a
    modelling decision without a number (an ideal LED model) has none.
    """

    key: str
    description: str
    value: Any = None
    unit: str | None = None

    def text(self) -> str:
        if self.value is None:
            return f"{self.key}: {self.description}"
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.key} = {_fmt(self.value)}{unit} - {self.description}"


class DesignChange(BaseModel):
    """One change a template asks for; the circuit agent turns it into an ``IRProposal`` (the design package does not import the agents)."""

    description: str
    target: str
    operation: str
    payload: Any = None
    rationale: str = ""


@dataclass
class Plan:
    """What a template would do, or why it does nothing.

    A buildable plan (``changes`` non-empty) carries the table the user must
    confirm: the inputs it read, the choices it makes, the numbers it
    computed and the parts it instantiates. A refusal carries ``notes`` and
    possibly non-required ``questions``; a template missing an input carries
    required ``questions``.
    """

    template: str
    title: str = ""
    version: str = TEMPLATE_VERSION
    changes: list[DesignChange] = field(default_factory=list)
    questions: list[MissingInformation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    inputs: dict[str, DesignInput] = field(default_factory=dict)
    choices: list[Choice] = field(default_factory=list)
    #: (parameter key, traced value) of every calculator output
    computed: list[tuple[str, Traced]] = field(default_factory=list)
    #: one line per part, net and simulation item, for the table
    parts: list[str] = field(default_factory=list)
    nets: list[str] = field(default_factory=list)
    simulation: list[str] = field(default_factory=list)
    #: one line per board decision: the stack and every signal-integrity class / timing path (:mod:`ai_eda.design.board`)
    board: list[str] = field(default_factory=list)
    #: keep-out areas the template's board needs (the transceiver's antenna band): :func:`~ai_eda.design.board.add_board` writes them into
    #: ``ir.pcb.keepouts`` with the board stack and lists them in the table; empty for every other template (its board is unchanged)
    keepouts: list[Keepout] = field(default_factory=list)
    #: one line per requirement the user left out of the design (:func:`left_out_lines`), shown in the table (and so part of its
    #: hash: a leave-out changes the table the user confirms); empty when nothing was left out (the table is unchanged)
    left_out: list[str] = field(default_factory=list)
    #: one line per requirement read under an alias of an input beside the one the input names (:func:`alias_duplicates`:
    #: ``req.battery_voltage`` beside ``req.input_voltage``, the same number), shown under the inputs; empty for every plan without one
    also_read: list[str] = field(default_factory=list)

    @property
    def buildable(self) -> bool:
        return bool(self.changes)

    def table(self) -> str:
        """The confirmation table: everything the user is asked to make theirs."""
        lines = [
            f"Template '{self.template}' v{self.version} ({self.title}) can be built from the confirmed requirements below. "
            f"Nothing is applied until you answer {CONFIRM_DESIGN_KEY}=yes ({CONFIRM_DESIGN_KEY}=no leaves the design empty).",
            "Inputs read (requirement -> value):",
        ]
        for key, inp in self.inputs.items():
            raw = inp.requirement.value.value if inp.requirement.value is not None else None
            lines.append(f"  {inp.requirement.id}: {key} = {_fmt(inp.traced.value)} {inp.traced.unit} (stated as {raw!r})")
        lines.extend(f"  {x}" for x in self.also_read)
        lines.append("Design choices the template makes (derived from nothing; confirming makes them your values):")
        lines.extend(f"  {c.text()}" for c in self.choices)
        lines.append("Computed by registered calculators (re-derived by the CALCULATION stage):")
        for key, t in self.computed:
            unit = f" {t.unit}" if t.unit else ""
            lines.append(f"  {key} = {_fmt(t.value)}{unit} [{t.provenance.tool} from {', '.join(t.provenance.derived_from)}]")
        lines.append("Parts instantiated from the KiCad library on disk:")
        lines.extend(f"  {p}" for p in self.parts)
        lines.append("Nets:")
        lines.extend(f"  {n}" for n in self.nets)
        lines.append("Simulation:")
        lines.extend(f"  {s}" for s in self.simulation)
        if self.board:
            lines.append("Board stack and signal integrity (need-driven: declared interfaces, and nets the routed copper shows to be electrically long):")
            lines.extend(f"  {b}" for b in self.board)
        if self.left_out:
            lines.append(f"Requirements you left out of this design ({LEAVE_OUT_KEY}; the template neither reads nor serves them and the reviewer does not "
                         f"demand them - a typed answer to one of their keys brings it back):")
            lines.extend(f"  {x}" for x in self.left_out)
        return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return repr(value) if isinstance(value, str) else str(value)
    return f"{value:.12g}"


@dataclass(frozen=True)
class TheorySection:
    """One section of a template's theory text: a title and a Markdown body (formulas as plain Unicode text, the IR's numbers substituted)."""

    title: str
    body: str


@dataclass(frozen=True)
class PartNote:
    """What a template says about one of its parts for the parts report.

    ``role``: what the part does in this circuit; ``why``: why this part class,
    value and footprint; ``criteria``: what any substitute must satisfy,
    computed from the design; ``substitutes``: candidate names the pipeline
    did **not** verify - each line carries :data:`UNVERIFIED_SUBSTITUTE`.
    """

    role: str
    why: str
    criteria: list[str] = field(default_factory=list)
    substitutes: list[str] = field(default_factory=list)


def parameter_value(ir: CircuitIR, key: str) -> float | None:
    """The numeric value of ``ir.parameters[key]``, or ``None`` when the key is missing or not a number (a report then prints :data:`NO_RECORD`)."""
    t = ir.parameters.get(key)
    if t is None or isinstance(t.value, bool) or not isinstance(t.value, (int, float)):
        return None
    return float(t.value)


def quantity(value: float | None, unit: str | None = None, digits: int = 5) -> str:
    """``value`` with an SI prefix and the unit's symbol (``6.48e-08 F`` -> ``64.817 nF``); :data:`NO_RECORD` for ``None``.

    Values in [0.1, 1000) keep no prefix (``0.7 V``, ``500 Hz``); outside that
    range the engineering exponent nearest below is used, down to pico and up
    to giga. A unitless number is printed as ``{digits}`` significant digits.
    """
    if value is None:
        return NO_RECORD
    u = UNIT_DISPLAY.get(unit or "", unit or "")
    if value == 0 or not math.isfinite(value) or not u:
        text = f"{value:.{digits}g}"
        return f"{text} {u}" if u else text
    if 0.1 <= abs(value) < 1000:
        return f"{value:.{digits}g} {u}"
    e3 = int(math.floor(math.log10(abs(value)) / 3.0)) * 3
    e3 = max(-12, min(9, e3))
    return f"{value / 10 ** e3:.{digits}g} {_SI_PREFIXES[e3]}{u}"


def number(value: float | None, digits: int = 6) -> str:
    """A bare number for a formula line (``{digits}`` significant digits); :data:`NO_RECORD` for ``None``."""
    return NO_RECORD if value is None else f"{value:.{digits}g}"


def unverified(name: str, note: str = "") -> str:
    """One substitute line: the candidate, an optional note and the marker that the pipeline verified nothing about it."""
    return f"{name}{' - ' + note if note else ''} {UNVERIFIED_SUBSTITUTE}"


@dataclass(frozen=True)
class LayerPolicy:
    """The board layer counts a template builds (``pcb_layers``) and the count it uses when no requirement states one.

    ``allowed`` holds only counts a generic stack exists for
    (:data:`~ai_eda.design.inputs.LAYER_COUNT_OPTIONS`) and contains
    ``default``. A policy that allows fewer counts than the generic stacks
    (:attr:`restricts`) must say why in ``reason``: the refusal of a stated
    count outside it, and the confirmation table row of its default, quote
    that sentence (``"the RF lines need a reference plane"``). Invalid
    policies raise ``ValueError`` when the template class is defined, never
    at run time.
    """

    allowed: tuple[int, ...] = LAYER_COUNT_OPTIONS
    default: int = DEFAULT_LAYER_COUNT
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.allowed:
            raise ValueError("a layer policy allows at least one layer count")
        for n in (*self.allowed, self.default):
            if isinstance(n, bool) or not isinstance(n, int):
                raise ValueError(f"a layer count is a plain integer, got {n!r}")
        unknown = [n for n in self.allowed if n not in LAYER_COUNT_OPTIONS]
        if unknown:
            raise ValueError(f"layer count(s) {unknown} have no generic stack (one of {list(LAYER_COUNT_OPTIONS)})")
        if len(set(self.allowed)) != len(self.allowed):
            raise ValueError(f"layer counts {list(self.allowed)} repeat a count")
        if self.default not in self.allowed:
            raise ValueError(f"the default {self.default} layers is not one of the allowed counts {list(self.allowed)}")
        if self.restricts and not self.reason.strip():
            raise ValueError(f"a policy that builds only {list(self.allowed)} layers must say why (reason)")

    @property
    def restricts(self) -> bool:
        """Whether the template builds fewer counts than the generic stacks (a design need, e.g. a reference plane)."""
        return set(self.allowed) != set(LAYER_COUNT_OPTIONS)

    @property
    def is_generic(self) -> bool:
        """Whether this is the policy every template had before policies existed: every generic stack, the global default."""
        return not self.restricts and self.default == DEFAULT_LAYER_COUNT

    def allowed_text(self) -> str:
        return " or ".join(str(n) for n in self.allowed)


#: every generic stack, 2 layers when no requirement states a count: the policy of every template that declares none
DEFAULT_LAYER_POLICY = LayerPolicy()


class Template(ABC):
    """A verified circuit template: which confirmed requirement keys select it, which it needs, serves and may ignore."""

    #: template id (``Provenance.tool`` suffix)
    id: str
    title: str
    version: str = TEMPLATE_VERSION
    #: canonical keys that select the template; ``all_triggers`` says whether every one of them is needed to trigger
    triggers: tuple[str, ...]
    all_triggers: bool = True
    #: canonical keys the template must have (missing ones are asked as required questions). A numeric key counts as present
    #: when :func:`~ai_eda.design.inputs.read_inputs` reads it; a categorical one (``modulation``, ``radio_build``) only when
    #: its reader reads it - :func:`~ai_eda.design.inputs.present_keys` is the one rule, for the selection gate (whose
    #: closed-world refusals run only once nothing is missing) and for the template's own ``build``
    needs: tuple[str, ...]
    #: canonical keys the template's design serves (the closed world: any other confirmed design requirement refuses it)
    serves: tuple[str, ...]
    #: canonical keys the template may leave unserved without refusing (beside :data:`IGNORED_KEYS`)
    ignores: tuple[str, ...] = ()
    #: the nets of a 4-layer board's planes: ``In1.Cu`` (ground) and ``In2.Cu`` (a supply; ``None``: ground again)
    plane_nets: tuple[str, str | None] = ("GND", None)
    #: the board layer counts this template builds and its default (read by :mod:`ai_eda.design.board`)
    layer_policy: LayerPolicy = DEFAULT_LAYER_POLICY

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        needs = cls.__dict__.get("needs")
        if isinstance(needs, tuple):
            unknown = [k for k in needs if k not in _NEEDABLE_KEYS]
            if unknown:
                # a key no reader can supply would never count as present, and a missing need switches the closed-world refusals off
                raise TypeError(f"template {cls.__name__}: needs {unknown}, which neither read_inputs nor a categorical reader supplies")

    def triggered(self, inputs: dict[str, DesignInput]) -> bool:
        present = [k in inputs for k in self.triggers]
        return all(present) if self.all_triggers else any(present)

    def triggered_by(self, ir: CircuitIR, inputs: dict[str, DesignInput]) -> bool:
        """Whether the confirmed requirements of ``ir`` select this template (``inputs``: :func:`~ai_eda.design.inputs.read_inputs` of it).

        The default is :meth:`triggered` on the numeric inputs. A template
        selected by a categorical requirement (``radio_build``, read by
        :func:`~ai_eda.design.inputs.read_radio_build`) overrides this and
        reads it from ``ir``; it never reads an unconfirmed value.
        """
        return self.triggered(inputs)

    def refusals(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str]) -> list[MissingInformation]:
        """One non-required question per confirmed design requirement this template cannot serve (closed world); empty when it may build.

        The generic rule: every requirement in a :data:`DESIGN_CATEGORIES`
        category whose value does not need verification must be a key the
        template serves or may ignore. A template with a validity condition
        on a served key (the divider's ``output_current``) adds its own.
        Each question's ``rationale`` is the short reason (the stage note).

        Every such question names the exact answer that leaves the
        requirement out (:func:`leave_out_remedy`, ``--answer leave_out=<key>``):
        a closed-world refusal is always leavable by an answer, never only by
        editing the IR.
        """
        out: list[MissingInformation] = []
        unserved = unserved_requirements(ir, self)
        keys = [r.key for r in unserved]
        for r in unserved:
            why = f"{r.id} ({requirement_text(r)}) is not served by the {self.title} template, which serves only {', '.join(self.serves)}"
            out.append(refusal_question(
                r.key,
                f"{why}; no template design was proposed. {leave_out_remedy(r.key, keys)}; or provide the circuit (components / nets) in the IR yourself.",
                why,
            ))
        return out

    @abstractmethod
    def build(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str], library: KicadLibrary, *, confirmed: bool) -> Plan: ...

    def theory(self, ir: CircuitIR) -> list[TheorySection]:
        """The circuit theory of this template with ``ir``'s numbers substituted (a view: nothing is written to the IR).

        Every template overrides this; the default says the template gives no theory text.
        """
        return [TheorySection("이론 설명 없음", f"템플릿 '{self.id}' v{self.version}은 이론 설명을 제공하지 않습니다.")]

    def theory_figures(self, ir: CircuitIR) -> list[Figure]:
        """The theory curves of this template drawn with ``ir``'s numbers (a view: nothing is written to the IR).

        Each figure's caption names the equation the theory text uses. A
        template whose numbers the IR lacks returns no figure for that curve
        (the report says so) instead of guessing. Default: no figure.
        """
        return []

    def theory_figure_vectors(self) -> tuple[str, ...]:
        """SPICE vectors (spelled as the IR spells them, ``v(OUT)``) the theory report plots beside the expectations' own; default none."""
        return ()

    def part_notes(self, ir: CircuitIR) -> dict[str, PartNote]:
        """Per reference designator: role, reason, substitute criteria and unverified candidates. Default: nothing (the parts report says so per part)."""
        return {}

    def si_declarations(self, ir: CircuitIR, ctx: BoardContext) -> SIDeclarations:
        """The signal-integrity classes / timing paths this circuit needs beyond the default classes (:mod:`ai_eda.design.board`).

        Default: none - every net is in the ``DEFAULT`` class and only the
        critical-length rule can promote one. ``ctx.choice`` adds a choice to
        the confirmation table; ``ctx.computed_param`` a calculator output.
        """
        from ai_eda.design.board import SIDeclarations

        return SIDeclarations()


def requirement_text(r: Requirement) -> str:
    v = r.value.value if r.value is not None else None
    return f"{r.key}: {v!r}" if v is not None else r.key


def specific_keys(bare_key: str) -> list[str]:
    """The canonical keys an ambiguous bare key could mean, computed now: every ``*_<bare_key>`` key of the same unit (``frequency`` -> ``clock_frequency``, ...).

    Read from :data:`~ai_eda.design.inputs.UNIT_OF` at call time, so a key
    added to it later (an RF ``carrier_frequency``) is listed too.
    """
    unit = AMBIGUOUS_KEYS.get(bare_key)
    return sorted(k for k, u in UNIT_OF.items() if u == unit and k.endswith(f"_{bare_key}"))


def served_through_specific_key(ir: CircuitIR, template: Template) -> dict[str, str]:
    """Confirmed requirements under an :data:`AMBIGUOUS_KEYS` key that the template serves after all: requirement id -> the canonical key.

    One counts as served only when the template serves a canonical key of
    the same unit whose confirmed value (:func:`~ai_eda.design.inputs.read_inputs`)
    reads exactly the same number as the ambiguous one
    (:func:`~ai_eda.design.inputs.read_value`): the user named the specific
    quantity with the same value, so nothing is guessed. Any other value, an
    unreadable one or no such key served leaves it unserved (closed world).
    """
    ambiguous = [
        r for r in ir.requirements.requirements
        if r.key in AMBIGUOUS_KEYS and canonical_key(r.key) is None and r.value is not None and r.value.provenance.is_authoritative
    ]
    if not ambiguous:
        return {}
    inputs, _ = read_inputs(ir)
    out: dict[str, str] = {}
    for r in ambiguous:
        unit = AMBIGUOUS_KEYS[r.key]
        traced, _why = read_value(r, unit)
        if traced is None:
            continue
        for canon in template.serves:
            if UNIT_OF.get(canon) == unit and canon in inputs and inputs[canon].traced.value == traced.value:
                out[r.id] = canon
                break
    return out


def unserved_requirements(ir: CircuitIR, template: Template) -> list[Requirement]:
    """Confirmed design-category requirements the template neither serves nor may ignore (an unconfirmed value is not counted).

    A board key (:data:`BOARD_KEYS`, the layer count) is served by every
    template through the stackup, so it never refuses one. A requirement
    under an ambiguous bare key (:data:`AMBIGUOUS_KEYS`, ``frequency``) is no
    alias of anything and is unserved - unless
    :func:`served_through_specific_key` finds the served specific key stated
    with the same number.
    """
    through = served_through_specific_key(ir, template)
    out: list[Requirement] = []
    for r in ir.requirements.requirements:
        if r.category not in DESIGN_CATEGORIES or r.key in IGNORED_KEYS:
            continue
        if r.value is not None and r.value.provenance.needs_verification:
            continue
        canon = canonical_key(r.key) or r.key
        if canon in template.serves or canon in template.ignores or canon in BOARD_KEYS or r.id in through:
            continue
        out.append(r)
    return out


__all__ = [
    "AMBIGUOUS_KEYS",
    "BOARD_KEYS",
    "CHOICE_NOTE_PREFIX",
    "CONFIRM_DESIGN_KEY",
    "DEFAULT_LAYER_POLICY",
    "DESIGN_CATEGORIES",
    "IGNORED_KEYS",
    "NO_RECORD",
    "TEMPLATE_VERSION",
    "TOOL_ID",
    "UNIT_DISPLAY",
    "UNVERIFIED_SUBSTITUTE",
    "Choice",
    "DesignChange",
    "LEAVE_OUT_KEY",
    "LayerPolicy",
    "PartNote",
    "Plan",
    "Template",
    "TheorySection",
    "choice_provenance",
    "leave_out_answer",
    "alias_duplicates",
    "design_references",
    "design_view_outside_requirements",
    "leavable_keys",
    "leave_out_remedy",
    "left_out_lines",
    "refusal_question",
    "number",
    "parameter_value",
    "quantity",
    "requirement_text",
    "served_through_specific_key",
    "specific_keys",
    "structural_provenance",
    "template_reads_key",
    "unusable_note",
    "unusable_remedy",
    "with_remedy",
    "template_tool",
    "unserved_requirements",
    "unverified",
]
