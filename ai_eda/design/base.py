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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from ai_eda.ir import CircuitIR, MissingInformation, Provenance, ProvenanceKind, Requirement, Traced
from ai_eda.tools.kicad.library import KicadLibrary

from ai_eda.design.inputs import DesignInput, canonical_key

TEMPLATE_VERSION = "0.1"
#: tool id of the template machinery (``Provenance.tool`` is ``design.template.<template id>`` on structural decisions)
TOOL_ID = "design.template"
#: how a confirmed free choice's note starts (the rest names the template, its version and the choice)
CHOICE_NOTE_PREFIX = "design choice confirmed by user"
#: the answer key that confirms a presented plan; a control key (never a requirement), see ``ai_eda.agents.requirement.CONTROL_KEYS``
CONFIRM_DESIGN_KEY = "confirm_design"

#: requirement categories a template must serve or refuse (the reviewer's ``requirements_vs_ir`` design categories)
DESIGN_CATEGORIES: frozenset[str] = frozenset({"electrical", "thermal", "mechanical", "signal_integrity", "power_integrity", "rf"})
#: requirement keys every template may ignore: they describe the product, not the circuit
IGNORED_KEYS: frozenset[str] = frozenset({"application", "jurisdiction"})


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
        return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return repr(value) if isinstance(value, str) else str(value)
    return f"{value:.12g}"


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


def requirement_text(r: Requirement) -> str:
    v = r.value.value if r.value is not None else None
    return f"{r.key}: {v!r}" if v is not None else r.key


def unserved_requirements(ir: CircuitIR, template: Template) -> list[Requirement]:
    """Confirmed design-category requirements the template neither serves nor may ignore (an unconfirmed value is not counted)."""
    out: list[Requirement] = []
    for r in ir.requirements.requirements:
        if r.category not in DESIGN_CATEGORIES or r.key in IGNORED_KEYS:
            continue
        if r.value is not None and r.value.provenance.needs_verification:
            continue
        canon = canonical_key(r.key) or r.key
        if canon in template.serves or canon in template.ignores:
            continue
        out.append(r)
    return out


__all__ = [
    "CHOICE_NOTE_PREFIX",
    "CONFIRM_DESIGN_KEY",
    "DESIGN_CATEGORIES",
    "IGNORED_KEYS",
    "TEMPLATE_VERSION",
    "TOOL_ID",
    "Choice",
    "DesignChange",
    "Plan",
    "Template",
    "choice_provenance",
    "requirement_text",
    "structural_provenance",
    "template_tool",
    "unserved_requirements",
]
