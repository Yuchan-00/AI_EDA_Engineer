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
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ai_eda.ir import CircuitIR, MissingInformation, Provenance, ProvenanceKind, Requirement, Traced
from ai_eda.tools.kicad.library import KicadLibrary

from ai_eda.design.inputs import BOARD_KEY_ALIASES, UNIT_OF, DesignInput, canonical_key, read_inputs, read_value

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
#: bare requirement keys that name a physical quantity but not *which* one (bare key -> its unit): never an alias of a
#: template input, so no template reads them; the circuit agent asks which specific key is meant (:func:`specific_keys`)
#: and a template counts one as served only beside the specific key stated with the same number
#: (:func:`served_through_specific_key`). ``frequency`` could be an oscillator's output, a filter corner, an MCU clock
#: or an RF carrier - reading it as any one of them would be a guess
AMBIGUOUS_KEYS: dict[str, str] = {"frequency": "Hz"}


def template_tool(template_id: str) -> str:
    return f"{TOOL_ID}.{template_id}"


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


class Template(ABC):
    """A verified circuit template: which confirmed requirement keys select it, which it needs, serves and may ignore."""

    #: template id (``Provenance.tool`` suffix)
    id: str
    title: str
    version: str = TEMPLATE_VERSION
    #: canonical keys that select the template; ``all_triggers`` says whether every one of them is needed to trigger
    triggers: tuple[str, ...]
    all_triggers: bool = True
    #: canonical keys the template must have (missing ones are asked as required questions)
    needs: tuple[str, ...]
    #: canonical keys the template's design serves (the closed world: any other confirmed design requirement refuses it)
    serves: tuple[str, ...]
    #: canonical keys the template may leave unserved without refusing (beside :data:`IGNORED_KEYS`)
    ignores: tuple[str, ...] = ()
    #: the nets of a 4-layer board's planes: ``In1.Cu`` (ground) and ``In2.Cu`` (a supply; ``None``: ground again)
    plane_nets: tuple[str, str | None] = ("GND", None)

    def triggered(self, inputs: dict[str, DesignInput]) -> bool:
        present = [k in inputs for k in self.triggers]
        return all(present) if self.all_triggers else any(present)

    def refusals(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str]) -> list[MissingInformation]:
        """One non-required question per confirmed design requirement this template cannot serve (closed world); empty when it may build.

        The generic rule: every requirement in a :data:`DESIGN_CATEGORIES`
        category whose value does not need verification must be a key the
        template serves or may ignore. A template with a validity condition
        on a served key (the divider's ``output_current``) adds its own.
        Each question's ``rationale`` is the short reason (the stage note).
        """
        out: list[MissingInformation] = []
        for r in unserved_requirements(ir, self):
            why = f"{r.id} ({requirement_text(r)}) is not served by the {self.title} template, which serves only {', '.join(self.serves)}"
            out.append(MissingInformation(
                key=r.key, required=False,
                question=f"{why}; no template design was proposed. Provide the circuit (components / nets) in the IR yourself, or leave this requirement out of a template design.",
                rationale=why,
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
    "DESIGN_CATEGORIES",
    "IGNORED_KEYS",
    "NO_RECORD",
    "TEMPLATE_VERSION",
    "TOOL_ID",
    "UNIT_DISPLAY",
    "UNVERIFIED_SUBSTITUTE",
    "Choice",
    "DesignChange",
    "PartNote",
    "Plan",
    "Template",
    "TheorySection",
    "choice_provenance",
    "number",
    "parameter_value",
    "quantity",
    "requirement_text",
    "served_through_specific_key",
    "specific_keys",
    "structural_provenance",
    "template_tool",
    "unserved_requirements",
    "unverified",
]
