"""The design file of ``ai-eda run --design FILE.json``: a circuit the user writes, through the same confirmation as a template.

Invariants:

* A file only *proposes*. :func:`load_design_file` turns it into a
  :class:`DesignFilePlan`, which :class:`~ai_eda.agents.circuit.CircuitDesignAgent`
  takes down the template path: table, recorded hash, ``confirm_design=yes``
  on a later run, into an empty design only. The table carries the file's
  sha256, so an edited file is asked again.
* A file cannot state where a value comes from: the input models forbid
  unknown keys, ``provenance`` among them, and every value is stamped
  ``user_requirement`` with a note naming the file's hash.
* A file cannot state pins: symbols and footprints are given by name and
  resolved on disk (:func:`~ai_eda.design.library_parts.library_component`),
  and a net naming a pin the symbol does not have refuses the file. So does
  a footprint whose pads do not pair with the symbol's pins by the PCB
  compiler's rule (:func:`~ai_eda.compilers.pins.match_pads`: a pin without
  a pad, a numbered plated pad that is no pin), before anything is confirmed
  or simulated; a mounting pad ``MP`` of a symbol without an ``MP`` pin and
  an unnumbered pad are mechanical and pass.
* Every part the simulation sees names the requirements it serves
  (``serves_requirements``), so the reviewer can trace them; a part excluded
  from SPICE with a reason (a connector) may name none.
* A model card (``spice.model_card``, verbatim ``.model`` / ``.subckt`` text)
  is judged by the SPICE compiler's own rules when the file is loaded, and
  its sha256 is computed here, never read from the file. ``model_source``
  says where the card comes from (a URL, a file name and its hash): it is the
  user's statement, recorded in the card's provenance note and not fetched.
  A card without one is still simulated, but its note says so
  (:data:`~ai_eda.ir.simulation.MODEL_CARD_NO_SOURCE`) and the SPICE stage
  reports what would pass as NOT_VERIFIED.
* A resistor, capacitor or inductor value is one that parts are sold in (IEC 60063:
  E24 / E96 for R, E6 / E12 / E24 for C and L - the series the templates place).
  Another value refuses the file and the message names the nearest standard
  values, unless the part states ``value_source``: one line naming the real part
  that has that value (a 5 mohm shunt). It is the file's word - shown in the
  confirmation table, never opened.
* A part may state ``min_pcb_layers`` - the smallest copper layer count its
  manufacturer says it needs (a BGA's escape) - only with
  ``min_pcb_layers_source``: one line naming the manufacturer document and its
  page or section (a part fact is read from the datasheet; one without its
  source refuses the file). Both are the file's word, shown in the
  confirmation table and never opened; the IR keeps the count as
  ``Component.min_pcb_layers`` with the source in its provenance, and the PCB
  agent's layer escalation starts at it.
* A part may state its ``manufacturer`` and ``mpn`` (the maker's part
  number), each one line of printable text a spreadsheet would not execute.
  They are the file's word like every value: shown in the confirmation
  table, stamped ``user_requirement``, never opened or looked up - the IR
  keeps them as ``Component.manufacturer`` / ``Component.mpn`` and the BOM
  prints them in its ``Manufacturer`` / ``MPN`` columns, while its
  ``DatasheetHash`` cell stays empty and its ``Unverified`` column (the
  verification status, in a column of its own) names it until a datasheet
  grounds the part number. Without
  them the plan, its table and hash and the IR are as before they existed.
* An optional ``si`` block (``"si": {}`` is enough; the format stays
  ``ai-eda-design/1``) gives the board the signal-integrity classes a
  template's board gets (:func:`~ai_eda.design.board.si_choices` /
  :func:`~ai_eda.design.board.si_constraints`, the same code and table rows):
  the ``DEFAULT`` class with the driver edge ``si.t_rise``, ``si.r_drive``,
  ``si.c_load``, ``si.ringing_tol`` and ``si.critical_fraction`` as the
  template machinery's labelled choices, so the ``si.*`` checks (crosstalk
  included), ``spice.si`` and ``spice.xtalk`` run on the routed copper. The
  block may state any of them instead (``"t_rise": "2 ns"``; a value with its
  unit, a fraction as a plain number in (0, 1]) - the user's value with the
  file's note - and ``xtalk_budget_rel``, the user's crosstalk budget (it
  takes precedence over the NM / 3 budget from grounded datasheet levels).
  **Validation only** unless the block says ``"promote": true``: the
  ``DEFAULT`` class promotes nothing and there is no controlled class, so no
  net gains a routing rule and the board is routed exactly as without the
  block - an electrically long net is reported by the SI checks, not routed
  as controlled impedance. ``promote`` adds the controlled class ``Z50`` and
  the promotion of electrically long nets, as on a template's board. Without
  the block the plan, its table and hash and the IR are as before it existed.
* Optional ``simulation.elements`` and ``simulation.nodes`` state what the
  netlist needs that the board does not have: simulation-only elements
  (:class:`~ai_eda.ir.simulation.SimElement`: a value R / C / L - a
  speaker's load across its connector -, a diode on a model card, or a
  voltage-controlled switch ``S`` with its two control nodes) between nets of
  the file or private simulation nodes, which are no net of the board and
  never reach the BOM, the schematic or the placement. A push button stays an
  excluded part; a switch across its nets, its control a PWL stimulus on a
  private node, closes and opens it inside one transient. They are judged at
  load by the SPICE compiler's rules for them, each refusal naming the field
  (``simulation.elements.2.control_net``): a device a simulation element may
  be, nodes that are nets of the file or listed nodes (plain names, none a
  net's), different ends, a switch's control pair (both, different; refused
  on every other device), a positive value in the device's unit for R / C / L
  and a model for D / S - defined by a card the netlist carries (an
  element's, or a part's that is not excluded from SPICE), one card per
  model name, a switch's an ``SW`` card stating exactly RON, ROFF, VT and VH
  (:func:`~ai_eda.compilers.spice.switch_model_problem`), no params on a
  switch -, params the compiler can write (plain names, a number or a single
  token each), an id that is no part's ref and no other netlist element's
  name, and a one-line ``reason``. An element's card follows a
  part's card rule above (judged and hashed here, ``model_source`` the
  user's word). Every value is the file's, like every other. Without them the
  setup, the table and its hash and the IR are as before they existed.
* An optional ``board`` section states a board layout the way a template
  proposes one (:mod:`ai_eda.design.board`): the ``outline``, ``fixed`` parts,
  ``pack_regions`` and ``silk_texts`` (mm and degrees, board frame, Y down),
  each one line of the table (:func:`~ai_eda.design.board.layout_lines`), into
  ``ir.pcb`` as a template's layout goes (:func:`~ai_eda.design.board.layout_fields`;
  appended to a board the IR already has, never replacing anything there:
  :func:`~ai_eda.design.board.layout_onto`). It brings no stack and no SI
  classes - those are the ``si`` block's and the PCB agent's - so a file
  that states a layout is routed and judged as one without it. The
  ``placement.fixed`` placer keeps the fixed parts where they are and packs
  each region's parts in its order; the silk texts, stamped as every value of
  the file (the mark the PCB agent keeps them by:
  :func:`~ai_eda.design.board.design_file_silk_text`), stay where they are and
  the silkscreen placer places the references and the title around them.
  Refused at load, naming the field: a ref no part of the file has, a part
  fixed twice, both fixed and packed or packed twice, fixed parts, regions or
  texts without an outline (they are positions on it; the placer places on
  one), a region or a text's anchor outside the outline, a text that is not
  one line of printable ASCII (the silkscreen placer's text-box estimate
  assumes Latin glyphs) or that the PCB compiler would refuse
  (:func:`~ai_eda.tools.silkscreen.geometry.silk_text_problems`).
  Without it the plan, its table and hash and the IR are as before it
  existed.
* A file that cannot be built is refused whole, before the pipeline starts
  (:class:`DesignFileError`; the CLI exits 2). The rules on what may reach
  ngspice stay the SPICE compiler's.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from ai_eda.compilers.pins import match_pads
from ai_eda.compilers.schematic import _REF_RE
from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.compilers.spice import (
    _BAD_NODE_CHARS,
    _NAME,
    _Ledger,
    _param_tokens,
    check_model_card,
    element_name,
    model_card_hash,
    switch_model_problem,
)
from ai_eda.design.base import CONFIRM_DESIGN_KEY, DESIGN_FILE_NOTE, Choice, DesignChange, Plan
from ai_eda.design.board import (
    CONTROLLED_CLASS,
    BoardContext,
    SIDeclarations,
    _layout_summary,
    class_line,
    design_file_layout_problem,
    has_layout,
    layout_fields,
    layout_lines,
    layout_onto,
    si_choices,
    si_constraints,
)
from ai_eda.design.inputs import LAYER_UNIT
from ai_eda.design.library_parts import TemplateRefusal, library_component
from ai_eda.design.templates import _net_line, _part_line
from ai_eda.errors import CompileError
from ai_eda.ir import (
    AnalysisSpec,
    Block,
    BoardOutline,
    BoardSide,
    CircuitDomain,
    Component,
    Expectation,
    Net,
    NetKind,
    PackRegion,
    PCBDesign,
    PinRef,
    PinAssociation,
    Placement,
    Provenance,
    ProvenanceKind,
    Reduce,
    SilkKind,
    SilkText,
    SimElement,
    SimulationSetup,
    SourceRef,
    SpiceBinding,
    SpiceDevice,
    Stimulus,
    StimulusKind,
    Topology,
    Traced,
    user_requirement,
)
from ai_eda.ir.pcb import _KEEPOUT_ID_RE
from ai_eda.ir.simulation import CONTROLLED_DEVICES, ELECTRICAL_KEYS, MODEL_CARD_NO_SOURCE, SIM_ELEMENT_DEVICES, VALUE_DEVICES
from ai_eda.tools.calc.e_series import is_standard, nearest
from ai_eda.tools.calc.quantity import parse_answer, parse_unit, unit_key
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.manufacturing.csv_cells import unsafe_cell
from ai_eda.tools.functional_intent import validate_intent
from ai_eda.tools.silkscreen.geometry import silk_text_problems
from ai_eda.tools.silkscreen.place import stroke_for
from ai_eda.tools.spice.ngspice_shared import NODE_RE
from ai_eda.tools.spice.runner import SpiceAnalysis

DESIGN_FILE_FORMAT = "ai-eda-design/1"
DESIGN_FILE_TEMPLATE = "design_file"
DESIGN_FILE_VERSION = "1"
#: the note of a non-empty design given ``--design``: nothing from the file is applied
NEEDS_EMPTY_IR = "ignored: --design needs an empty IR"


class DesignFileError(Exception):
    """The design file cannot be read or built: one line for the user, nothing was run."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Quantity(_Strict):
    """A number with its unit (``{"value": 5, "unit": "V"}``); a bare number, string, bool or list is a value without one."""

    value: Any
    unit: str | None = None


Value = bool | int | float | str | list[Any] | Quantity


class DesignSpice(_Strict):
    device: SpiceDevice | None = None
    exclude: bool = False
    exclude_reason: str = ""
    value: Value | None = None
    model_name: str | None = None
    model_card: str | None = None
    #: where the card comes from (URL, or file name and hash), one line; without it nothing the card decides passes
    model_source: str | None = None
    #: pin numbers in SPICE node order; the symbol's own order when absent
    pin_order: list[str] | None = None
    ignored_pins: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Value] = Field(default_factory=dict)


class DesignPart(_Strict):
    ref: str
    value: str
    description: str = ""
    symbol: str  # Library:Name
    footprint: str  # Library:Name
    electrical: dict[str, Value] = Field(default_factory=dict)
    serves_requirements: list[str] = Field(default_factory=list)
    #: the real part whose value this is (part number or datasheet, one line) when an R / C / L value is outside the E series
    value_source: str | None = None
    #: the smallest copper layer count the part's manufacturer states it needs, and the document and page that state it (one line;
    #: required with the count)
    min_pcb_layers: int | None = None
    min_pcb_layers_source: str | None = None
    #: the maker and its part number as the file states them (one line each; module docstring): the BOM's Manufacturer / MPN
    manufacturer: str | None = None
    mpn: str | None = None
    spice: DesignSpice


class DesignNet(_Strict):
    name: str
    kind: NetKind = NetKind.SIGNAL
    pins: list[str]  # R1.1
    serves_requirements: list[str] = Field(default_factory=list)


class DesignBlock(_Strict):
    """The existing IR block fields, with provenance supplied by the file loader."""

    id: str = Field(min_length=1)
    function: str = Field(min_length=1)
    domain: CircuitDomain = CircuitDomain.OTHER
    component_refs: list[str] = Field(default_factory=list)
    input_nets: list[str] = Field(default_factory=list)
    output_nets: list[str] = Field(default_factory=list)


class DesignPinAssociation(_Strict):
    terminal: str  # support part terminal, e.g. C1.1
    ic_pin: str  # served IC pin, e.g. U1.10
    role: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class DesignTopology(_Strict):
    name: str
    domains: list[CircuitDomain] = Field(default_factory=list)
    rationale: str = ""
    blocks: list[DesignBlock] = Field(default_factory=list)
    pin_associations: list[DesignPinAssociation] = Field(default_factory=list)
    #: None: the textbook schematic layout (1.0) unless the run says otherwise; "legacy" selects the 0.8 symbol table
    schematic_layout: Literal["legacy", "textbook"] | None = None


class DesignStimulus(_Strict):
    id: str
    source: Literal["voltage", "current"]
    net: str
    reference_net: str
    kind: StimulusKind
    value: Value | None = None
    params: dict[str, Value] = Field(default_factory=dict)
    serves_requirements: list[str] = Field(default_factory=list)


class DesignAnalysis(_Strict):
    id: str
    kind: SpiceAnalysis
    params: dict[str, Value] = Field(default_factory=dict)


class DesignExpectation(_Strict):
    id: str
    analysis_id: str
    vector: str
    reduce: Reduce = Reduce.VALUE
    at: Value | None = None
    nominal: Value
    tol_abs: Value | None = None
    tol_rel: Value | None = None
    requirement_id: str | None = None
    params: dict[str, Value] = Field(default_factory=dict)
    reference_vector: str | None = None
    bound: Literal["at_least", "at_most"] | None = None


class DesignSimElement(_Strict):
    """A simulation-only element of ``simulation.elements`` (module docstring): the fields of :class:`~ai_eda.ir.simulation.SimElement`,
    provenance supplied by the file loader, and a card's ``model_source`` as on a part."""

    id: str
    device: SpiceDevice
    net: str
    reference_net: str
    #: the control node pair of a voltage-controlled switch (``device`` ``S``: both required, closed above VT + VH of
    #: v(control_net) - v(control_reference_net)); every other device refuses them
    control_net: str | None = None
    control_reference_net: str | None = None
    value: Value | None = None
    model_name: str | None = None
    model_card: str | None = None
    #: where the card comes from (URL, or file name and hash), one line; without it nothing the card decides passes
    model_source: str | None = None
    params: dict[str, Value] = Field(default_factory=dict)
    #: what the element stands for and why it is not a part, one line (the table and the reports print it)
    reason: str
    serves_requirements: list[str] = Field(default_factory=list)


class DesignSimulation(_Strict):
    stimuli: list[DesignStimulus] = Field(default_factory=list)
    analyses: list[DesignAnalysis] = Field(default_factory=list)
    expectations: list[DesignExpectation] = Field(default_factory=list)
    #: simulation-only elements (module docstring); empty: the setup is the one a file gave before they existed
    elements: list[DesignSimElement] = Field(default_factory=list)
    #: private simulation node names (a press schedule's node): usable by elements and stimuli, never a net of the board
    nodes: list[str] = Field(default_factory=list)


class DesignSI(_Strict):
    """The optional ``si`` block: the board's SI net classes (module docstring). ``{}`` takes every value from the template machinery's choices."""

    #: the driver edge / source resistance / far-end load, each with its unit (``"2 ns"`` or ``{"value": 2, "unit": "ns"}``)
    t_rise: Value | None = None
    r_drive: Value | None = None
    c_load: Value | None = None
    #: fractions (plain numbers in (0, 1]): the SPICE check's ringing band, the critical-length rule's fraction, the crosstalk budget
    ringing_tol: Value | None = None
    critical_fraction: Value | None = None
    xtalk_budget_rel: Value | None = None
    #: promote electrically long nets to the controlled class and route them again, as a template's board does (default: validation only)
    promote: StrictBool = False


#: the si block's dimensional keys and their unit
SI_UNITS: dict[str, str] = {"t_rise": "s", "r_drive": "ohm", "c_load": "F"}
#: the si block's fractions (no unit, in (0, 1])
SI_FRACTIONS: tuple[str, ...] = ("ringing_tol", "critical_fraction", "xtalk_budget_rel")


#: a number of the ``board`` section: a plain finite number (mm, or degrees for a rotation) - a string, a bool, inf and NaN are refused
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class DesignOutline(_Strict):
    """The board's rectangular outline (:class:`~ai_eda.ir.BoardOutline`), mm."""

    width_mm: Number
    height_mm: Number
    origin_x_mm: Number = 0.0
    origin_y_mm: Number = 0.0


class DesignFixed(_Strict):
    """A part the file fixes (:attr:`~ai_eda.ir.PCBDesign.fixed`): its footprint's position (mm), rotation (degrees) and side, kept as given."""

    ref: str
    x_mm: Number
    y_mm: Number
    rotation_deg: Number = 0.0
    side: BoardSide = BoardSide.TOP


class DesignPackRegion(_Strict):
    """A rectangle ``[x, y, w, h]`` (mm) the ``placement.fixed`` placer packs ``refs`` into, in their list order (:class:`~ai_eda.ir.PackRegion`)."""

    id: str
    rect_mm: list[Number]
    refs: list[str]
    side: BoardSide = BoardSide.TOP


class DesignSilkText(_Strict):
    """A board text on the silk layer of ``side`` (:class:`~ai_eda.ir.SilkText`, kind ``user`` or ``title``): its anchor (mm), glyph height
    (mm; the stroke is the silkscreen placer's own rule, :func:`~ai_eda.tools.silkscreen.place.stroke_for`), angle, justification and the
    part it labels."""

    text: str
    x_mm: Number
    y_mm: Number
    size_mm: Number = 1.0
    rotation_deg: Number = 0.0
    justify: Literal["center", "left", "right"] = "center"
    side: BoardSide = BoardSide.TOP
    kind: Literal["user", "title"] = "user"
    #: the part the text belongs to (a key's note name), when it labels one
    ref: str | None = None


class DesignBoard(_Strict):
    """The optional ``board`` section (module docstring): a board layout as a template proposes one - mm and degrees, board frame, Y down."""

    outline: DesignOutline | None = None
    fixed: list[DesignFixed] = Field(default_factory=list)
    pack_regions: list[DesignPackRegion] = Field(default_factory=list)
    silk_texts: list[DesignSilkText] = Field(default_factory=list)


class DesignFile(_Strict):
    format: Literal["ai-eda-design/1"]
    title: str
    topology: DesignTopology
    components: list[DesignPart] = Field(min_length=1)
    nets: list[DesignNet] = Field(min_length=1)
    simulation: DesignSimulation | None = None
    si: DesignSI | None = None
    board: DesignBoard | None = None


@dataclass
class DesignFilePlan(Plan):
    """A :class:`~ai_eda.design.base.Plan` whose table says it is a file's, with the file's hash in it."""

    sha256: str = ""
    topology_line: str = ""
    #: the head of the signal-integrity section (empty: the file has no ``si`` block and the table is as before it existed)
    si_line: str = ""
    intent_lines: list[str] = field(default_factory=list)
    #: one line per item of the file's ``board`` section (:func:`~ai_eda.design.board.layout_lines`; empty: the file has none and the
    #: table is as before it existed)
    layout: list[str] = field(default_factory=list)

    def for_board(self, pcb: PCBDesign | None) -> DesignFilePlan:
        """This plan for an IR whose board is ``pcb``: the plan itself, unless the file has a ``board`` section and the IR already has a board.

        Then, as :func:`~ai_eda.design.board.add_board` does for a template, the
        layout is appended to that board (:func:`~ai_eda.design.board.layout_onto`:
        the outline set, each fixed part, region and text appended, nothing
        replaced) - or, when it clashes with what the board holds, the copy
        proposes nothing and says why. The table is the same either way.
        """
        if pcb is None or not has_layout(self):
            return self
        appended, why = layout_onto(pcb, self, f"{DESIGN_FILE_NOTE}{self.sha256}; {_layout_summary(self)}", "the design file")
        if why is not None:
            return replace(self, changes=[], notes=[*self.notes, f"design file {self.sha256} not proposed: its board section clashes with the IR's board - {why}"])
        return replace(self, changes=[*(c for c in self.changes if c.target != "pcb"), *appended])

    def table(self) -> str:
        lines = [
            f"Design file '{self.title}' ({self.sha256}) can be built as written. Nothing is applied until you answer {CONFIRM_DESIGN_KEY}=yes "
            f"({CONFIRM_DESIGN_KEY}=no leaves the design empty). Confirming makes every value below your own (user_requirement); no value in it is checked yet.",
            f"Topology: {self.topology_line}",
            "Parts instantiated from the KiCad library on disk:",
            *(f"  {p}" for p in self.parts),
            "Nets:",
            *(f"  {n}" for n in self.nets),
            "Simulation:",
            *(f"  {s}" for s in self.simulation),
        ]
        if self.intent_lines:
            lines += ["Functional intent (declared by the file):", *(f"  {line}" for line in self.intent_lines)]
        if self.si_line:
            lines += [self.si_line, "SI design choices (confirming makes them your values; overrides are labelled as your stated values):",
                      *(f"  {c.text()}" for c in self.choices), *(f"  {b}" for b in self.board)]
        if self.layout:
            lines += ["Board layout (the file's board section, kept as written: a fixed part is never moved, a region's parts are packed into it in "
                      "its order, a silk text stays where it is and the silkscreen is placed around it):", *(f"  {b}" for b in self.layout)]
        return "\n".join(lines)


def _no_constant(name: str) -> None:
    raise ValueError(f"{name} is not a number a design file may state")


def _lib_id(text: str, what: str) -> tuple[str, str]:
    library, sep, name = text.partition(":")
    if not sep or not library or not name:
        raise DesignFileError(f"--design: {what} {text!r} is not Library:Name")
    return library, name


def _plain(value: Any) -> str:
    return f"{value.value}{' ' + value.unit if value.unit else ''}" if isinstance(value, Quantity) else str(value)


def _params_text(params: dict[str, Value]) -> str:
    return ", ".join(f"{k}={_plain(v)}" for k, v in params.items())


def _card_source(s: DesignSpice, ref: str) -> tuple[str, str]:
    """``(sha256 of the card as the netlist will carry it, its one-line source or "")``; refuses a card the SPICE compiler would refuse."""
    try:
        card, _ = check_model_card(ref, s.model_name, s.model_card or "")
    except CompileError as e:
        raise DesignFileError(f"--design: {e}") from e
    source = (s.model_source or "").strip()
    if s.model_source is not None and (not source or not source.isprintable() or MODEL_CARD_NO_SOURCE in source):
        raise DesignFileError(f"--design: {ref} model_source must be one line of text saying where the card comes from")
    return model_card_hash(card), source


def _element_card(x: DesignSimElement, where: str) -> tuple[str, str]:
    """:func:`_card_source` for a simulation element's card: the same rule, each refusal naming the field (``where``: its path)."""
    try:
        card, _ = check_model_card(f"simulation element {x.id}", x.model_name, x.model_card or "")
    except CompileError as e:
        raise DesignFileError(f"--design: {where}.model_card: {e}") from e
    source = (x.model_source or "").strip()
    if x.model_source is not None and (not source or not source.isprintable() or MODEL_CARD_NO_SOURCE in source):
        raise DesignFileError(f"--design: {where}.model_source: simulation element {x.id}: model_source must be one line of text saying where the card comes from")
    return model_card_hash(card), source


def _card_text(text: str, sha256: str, source: str) -> str:
    """The table's words for a model card: its text, its hash, and where the file says it comes from - or that it names no source."""
    return f"model_card {text!r} ({sha256}, " + (f"from {source} as you state it, not fetched)" if source else "no model_source: SPICE results that would pass are NOT_VERIFIED)")


def _spice_text(s: DesignSpice, ref: str) -> str:
    if s.exclude:
        return f"excluded from SPICE: {s.exclude_reason}"
    card = ""
    if s.model_card is not None:
        card = _card_text(s.model_card, *_card_source(s, ref))
    extras = [
        f"pin_order {s.pin_order}" if s.pin_order is not None else "",
        f"ignored_pins {s.ignored_pins}" if s.ignored_pins else "",
        card,
    ]
    return " ".join(x for x in (f"SPICE {s.device}", _plain(s.value) if s.value is not None else "", s.model_name or "", _params_text(s.params), *extras) if x)


#: the IEC 60063 series a resistor / capacitor / inductor value of a design file must come from (the same ones the templates place)
STANDARD_SERIES: dict[SpiceDevice, tuple[str, ...]] = {SpiceDevice.R: ("e24", "e96"), SpiceDevice.C: ("e6", "e12", "e24"), SpiceDevice.L: ("e6", "e12", "e24")}


#: the unit a resistor / capacitor / inductor value is stated in
DEVICE_UNITS: dict[SpiceDevice, str] = {SpiceDevice.R: "ohm", SpiceDevice.C: "F", SpiceDevice.L: "H"}


def _nonstandard_value(p: DesignPart) -> str | None:
    """Why ``p``'s R / C / L value is not one parts are sold in, or ``None`` (a standard value, a zero, no number, another device)."""
    series = STANDARD_SERIES.get(p.spice.device) if p.spice.device is not None else None
    raw = p.spice.value.value if isinstance(p.spice.value, Quantity) else p.spice.value
    if series is None or isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    stated = p.spice.value.unit if isinstance(p.spice.value, Quantity) else None
    if stated is not None and unit_key(stated) != unit_key(DEVICE_UNITS[p.spice.device]):  # 1000 "V" on a resistor is not 1 kohm
        return f"{p.ref} value is stated in {stated!r}, a {p.spice.device.value} takes {DEVICE_UNITS[p.spice.device]}"
    if raw <= 0:
        return None
    if any(is_standard(raw, s) for s in series):
        return None
    unit = p.spice.value.unit if isinstance(p.spice.value, Quantity) and p.spice.value.unit else ""
    near = sorted({nearest(raw, s) for s in (series[-2], series[-1])})
    return (
        f"{p.ref} value {raw:.12g} {unit}".rstrip() + f" is not a value parts are sold in (IEC 60063 {' / '.join(s.upper() for s in series)}); "
        f"nearest: {', '.join(f'{v:.12g}' for v in near)}"
    )


def _part_extras(p: DesignPart) -> str:
    """What the part line of the table adds for the file's own electrical values and traced requirements (empty when it states none)."""
    electrical = f"; electrical {_params_text(p.electrical)}" if p.electrical else ""
    serves = f"; serves {', '.join(p.serves_requirements)}" if p.serves_requirements else ""
    # the file's word for a value outside the E series: shown in the table the user confirms, never opened or checked
    source = f"; value outside the E series, stated part: {p.value_source.strip()}" if (p.value_source or "").strip() else ""
    layers = (f"; needs at least {p.min_pcb_layers} copper layers per {p.min_pcb_layers_source.strip()} (the file's word, not opened)"
              if p.min_pcb_layers is not None and p.min_pcb_layers_source else "")
    named = [f"{what} {text.strip()}" for what, text in (("manufacturer", p.manufacturer), ("part number", p.mpn)) if text is not None]
    identity = f"; {', '.join(named)} (the file's word, not looked up)" if named else ""
    return electrical + serves + source + layers + identity


def _identity_problem(p: DesignPart) -> str | None:
    """Why ``p``'s ``manufacturer`` / ``mpn`` cannot be taken (one line of printable text each, no cell a spreadsheet would execute -
    the BOM's rule), or ``None``."""
    for field_name, text in (("manufacturer", p.manufacturer), ("mpn", p.mpn)):
        if text is None:
            continue
        if not text.strip() or len(text.splitlines()) != 1 or not text.strip().isprintable():
            return f"{p.ref} {field_name} must be one line of printable text"
        why = unsafe_cell(text.strip())
        if why is not None:
            return f"{p.ref} {field_name} {text.strip()!r} {why}: the BOM refuses a cell spreadsheet software would execute"
    return None


def _min_layers_problem(p: DesignPart) -> str | None:
    """Why ``p``'s ``min_pcb_layers`` / ``min_pcb_layers_source`` cannot be taken, or ``None`` (neither stated, or both well formed)."""
    if p.min_pcb_layers is None and p.min_pcb_layers_source is None:
        return None
    if p.min_pcb_layers is None:
        return f"{p.ref} has min_pcb_layers_source without min_pcb_layers"
    if isinstance(p.min_pcb_layers, bool) or p.min_pcb_layers < 1:
        return f"{p.ref} min_pcb_layers must be a whole copper layer count >= 1, got {p.min_pcb_layers!r}"
    source = p.min_pcb_layers_source
    if source is None or not source.strip():
        return (f"{p.ref} min_pcb_layers {p.min_pcb_layers} needs min_pcb_layers_source: one line naming the manufacturer document and its page "
                "or section that states it (a part fact is read from the datasheet)")
    if len(source.splitlines()) != 1 or not source.strip().isprintable():
        return f"{p.ref} min_pcb_layers_source must be one line of text naming the manufacturer document and its page or section"
    return None


class _Builder:
    """Turns the file's blocks into the IR's models, every value stamped with the file's hash."""

    def __init__(self, sha256: str, library: KicadLibrary) -> None:
        self.note = f"{DESIGN_FILE_NOTE}{sha256}"
        self.library = library
        self.provenance = Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note=self.note)

    def traced(self, value: Value) -> Traced:
        if isinstance(value, Quantity):
            return user_requirement(value.value, value.unit, self.note)
        return user_requirement(value, None, self.note)

    def optional(self, value: Value | None) -> Traced | None:
        return None if value is None else self.traced(value)

    def params(self, params: dict[str, Value]) -> dict[str, Traced]:
        return {k: self.traced(v) for k, v in params.items()}

    def part(self, p: DesignPart) -> Component:
        c = library_component(self.library, p.ref, p.value, p.description, _lib_id(p.symbol, f"{p.ref} symbol"), _lib_id(p.footprint, f"{p.ref} footprint"), self.provenance, p.serves_requirements)
        self.check_pads(c, p)
        c.electrical = self.params(p.electrical)
        s = p.spice
        value = self.optional(s.value)
        key = ELECTRICAL_KEYS.get(s.device) if s.device is not None else None
        if key is not None and value is not None:
            c.electrical.setdefault(key, value)
        card = None
        if s.model_card is not None:
            sha256, source = _card_source(s, p.ref)
            card = user_requirement(s.model_card, None, f"{self.note}; model card {sha256} " + (f"from {source}" if source else MODEL_CARD_NO_SOURCE))
        elif s.model_source is not None:
            raise DesignFileError(f"--design: {p.ref} has model_source without model_card")
        c.spice = SpiceBinding(
            device=s.device, exclude=s.exclude, exclude_reason=s.exclude_reason, value=value, model_name=s.model_name, model_card=card,
            pin_order=s.pin_order if s.pin_order is not None else [x.number for x in c.pins], ignored_pins=s.ignored_pins, params=self.params(s.params), provenance=self.provenance,
        )
        for field_name, text in (("manufacturer", p.manufacturer), ("mpn", p.mpn)):  # checked by _identity_problem before any part is built
            if text is not None:
                setattr(c, field_name, user_requirement(text.strip(), None, f"{self.note}; as the file states it (not looked up)"))
        if p.min_pcb_layers is not None and p.min_pcb_layers_source is not None:  # checked by _min_layers_problem before any part is built
            source = p.min_pcb_layers_source.strip()
            c.min_pcb_layers = Traced(value=p.min_pcb_layers, unit=LAYER_UNIT, provenance=Provenance(
                kind=ProvenanceKind.USER_REQUIREMENT, source=SourceRef(title=source),
                note=f"{self.note}; the part needs at least {p.min_pcb_layers} copper layers per {source}, as the file states it (not opened)"))
        return c

    def check_pads(self, c: Component, p: DesignPart) -> None:
        """The PCB compiler's pad / pin rule for one part (module docstring), so the file is refused before it is confirmed."""
        try:
            fp = self.library.load_footprint(c.footprint)  # type: ignore[arg-type]
        except CompileError as e:  # LibraryFormatError: a malformed .kicad_mod
            raise DesignFileError(f"--design: {p.ref} footprint {p.footprint}: {e}") from e
        match = match_pads([x.number for x in c.pins], fp.pads)
        if match.without_pad:
            raise DesignFileError(f"--design: {p.ref}: symbol pins {sorted(match.without_pad, key=natural_ref_key)} of {p.symbol!r} have no pad in footprint "
                                  f"{p.footprint} (the PCB compiler refuses it)")
        if match.extra:
            raise DesignFileError(f"--design: {p.ref}: footprint {p.footprint} has pads {sorted(match.extra, key=natural_ref_key)} that are not pins of "
                                  f"{p.symbol!r} (the PCB compiler refuses it)")

    def net(self, n: DesignNet, parts: dict[str, Component]) -> Net:
        pins = []
        for text in n.pins:
            ref, sep, number = text.rpartition(".")
            if not sep or ref not in parts or parts[ref].pin(number) is None:
                have = f"its symbol has {[x.number for x in parts[ref].pins]}" if ref in parts else "no such part"
                raise DesignFileError(f"--design: net {n.name} names pin {text!r}: {have}")
            pins.append(PinRef(component_ref=ref, pin_number=number))
        return Net(name=n.name, kind=n.kind, pins=pins, provenance=self.provenance, serves_requirements=n.serves_requirements)

    def element(self, x: DesignSimElement, where: str) -> SimElement:
        """A simulation-only element (checked by :func:`_check_sim_elements`), its card noted as a part's is."""
        card = None
        if x.model_card is not None:
            sha256, source = _element_card(x, where)
            card = user_requirement(x.model_card, None, f"{self.note}; model card {sha256} " + (f"from {source}" if source else MODEL_CARD_NO_SOURCE))
        return SimElement(
            id=x.id, device=x.device, net=x.net, reference_net=x.reference_net, control_net=x.control_net, control_reference_net=x.control_reference_net,
            value=self.optional(x.value), model_name=x.model_name, model_card=card, params=self.params(x.params), reason=x.reason.strip(),
            provenance=self.provenance, serves_requirements=x.serves_requirements,
        )

    def simulation(self, s: DesignSimulation) -> SimulationSetup:
        return SimulationSetup(
            elements=[self.element(x, f"simulation.elements.{i}") for i, x in enumerate(s.elements)],
            nodes=list(s.nodes),
            stimuli=[
                Stimulus(id=x.id, source=x.source, net=x.net, reference_net=x.reference_net, kind=x.kind, value=self.optional(x.value), params=self.params(x.params), provenance=self.provenance, serves_requirements=x.serves_requirements)
                for x in s.stimuli
            ],
            analyses=[AnalysisSpec(id=x.id, kind=x.kind, params=self.params(x.params), provenance=self.provenance) for x in s.analyses],
            expectations=[
                Expectation(
                    id=x.id, analysis_id=x.analysis_id, vector=x.vector, reduce=x.reduce, at=self.optional(x.at), nominal=self.traced(x.nominal), tol_abs=self.optional(x.tol_abs), tol_rel=self.optional(x.tol_rel),
                    requirement_id=x.requirement_id, params=self.params(x.params), reference_vector=x.reference_vector, bound=x.bound, provenance=self.provenance,
                )
                for x in s.expectations
            ],
        )


def _element_lines(s: DesignSimulation) -> list[str]:
    """Table lines for the private nodes and the simulation-only elements (none for a file that states neither)."""
    out = [f"simulation-only node(s): {', '.join(s.nodes)} (no net of the board)"] if s.nodes else []
    for i, x in enumerate(s.elements):
        control = f", closed by v({x.control_net}) - v({x.control_reference_net}) above VT + VH" if x.device in CONTROLLED_DEVICES else ""
        body = _plain(x.value) if x.value is not None else f"model {x.model_name}"
        params = f" ({_params_text(x.params)})" if x.params else ""
        card = f" {_card_text(x.model_card, *_element_card(x, f'simulation.elements.{i}'))}" if x.model_card is not None else ""
        serves = f"; serves {', '.join(x.serves_requirements)}" if x.serves_requirements else ""
        out.append(f"simulation-only element {x.id}: {x.device.value} between {x.net} and {x.reference_net}{control}: {body}{params}{card} "
                   f"- {x.reason.strip()}{serves}")
    return out


def _simulation_lines(s: DesignSimulation) -> list[str]:
    out = _element_lines(s)
    out += [f"stimulus {x.id}: {x.source} {x.kind.value} on {x.net} against {x.reference_net}{' = ' + _plain(x.value) if x.value is not None else ''}{' (' + _params_text(x.params) + ')' if x.params else ''}" for x in s.stimuli]
    out += [f"analysis {x.id}: {x.kind.value} ({_params_text(x.params)})" for x in s.analyses]
    for x in s.expectations:
        limits = [f"{name} {_plain(v)}" for name, v in (("tol_abs", x.tol_abs), ("tol_rel", x.tol_rel)) if v is not None] + ([x.bound] if x.bound else [])
        at = f" at {_plain(x.at)}" if x.at is not None else ""
        verifies = f" verifies {x.requirement_id}" if x.requirement_id else ""
        reference = f" against {x.reference_vector}" if x.reference_vector else ""
        params = f" [{_params_text(x.params)}]" if x.params else ""
        out.append(f"expectation {x.id}: {x.reduce.value} of {x.vector}{reference} in {x.analysis_id}{at}{params} = {_plain(x.nominal)} ({', '.join(limits)}){verifies}")
    return out


def _duplicates(items: list[str]) -> list[str]:
    return sorted({x for x in items if items.count(x) > 1})


def _si_number(key: str, raw: Value) -> float:
    """The number the si block states under ``key``, in the IR's unit (``"2 ns"`` -> 2e-09); :class:`DesignFileError` naming why not.

    A dimensional value needs its unit, read by the requirement parser's rules
    (:func:`~ai_eda.tools.calc.quantity.parse_answer`; a bare number is not
    read - the unit is never guessed); a fraction is a plain number in (0, 1].
    """
    where = f"--design: si.{key}"
    unit = SI_UNITS.get(key)
    if unit is None:
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not 0 < raw <= 1:
            raise DesignFileError(f"{where} must be a plain number in (0, 1] (a fraction, no unit), got {_plain(raw)}")
        return float(raw)
    if isinstance(raw, str):
        q = parse_answer(raw)
        if q is None or q.unit != unit:
            raise DesignFileError(f"{where} {raw!r} is not one quantity in {unit} (e.g. \"2 ns\", \"33 ohm\", \"10 pF\")")
        value = q.value
    elif isinstance(raw, Quantity) and isinstance(raw.value, (int, float)) and not isinstance(raw.value, bool) and raw.unit is not None:
        parsed = parse_unit(raw.unit)
        if parsed is None or parsed[0] != unit:
            raise DesignFileError(f"{where} is stated in {raw.unit!r}, it takes {unit}")
        value = float(Decimal(str(raw.value)).scaleb(parsed[1]))  # decimal scaling agrees with the quantity parser
    else:
        raise DesignFileError(f"{where} needs its unit ({unit}: e.g. \"2 ns\" or {{\"value\": 2, \"unit\": \"ns\"}}); a number without one is not read")
    if not math.isfinite(value) or value <= 0:
        raise DesignFileError(f"{where} must be > 0, got {_plain(raw)}")
    return value


def _add_si(plan: DesignFilePlan, block: DesignSI, note: str) -> None:
    """The file's ``si`` block into ``plan``: the template machinery's SI classes and table rows (:mod:`ai_eda.design.board`).

    The plan reaches the IR only through ``confirm_design=yes``, so the
    choices are built as confirmed (``user_requirement``, the template
    machinery's choice note), like a template's on the run that applies it;
    a value the block states is the user's with the file's note. Without
    ``promote`` there is no controlled class and nothing is promoted:
    validation only (:data:`~ai_eda.design.board.VALIDATION_ONLY`).
    """
    given = block.model_dump(exclude_none=True, exclude={"promote"})
    numbers = {key: _si_number(key, getattr(block, key)) for key in given}
    stated = {f"si.{key}": user_requirement(v, SI_UNITS.get(key), note) for key, v in numbers.items() if key != "xtalk_budget_rel"}
    budget = user_requirement(numbers["xtalk_budget_rel"], None, note) if "xtalk_budget_rel" in numbers else None
    ctx = BoardContext(template_id=DESIGN_FILE_TEMPLATE, confirmed=True, params={}, stackup=None, driver={})
    fraction, z0, z0_tol = si_choices(ctx, promote=block.promote, stated=stated)
    si = si_constraints(ctx, SIDeclarations(), fraction, z0, z0_tol, xtalk_budget=budget)
    plan.choices = list(ctx.choices)
    if budget is not None:
        plan.choices.append(Choice("si.xtalk_budget_rel", "your crosstalk budget, a fraction of the victim's swing: it takes precedence over the NM / 3 budget "
                                   "spice.xtalk computes from grounded datasheet levels", budget.value))
    plan.board = [class_line(c) for c in si.net_classes]
    plan.si_line = ("Signal integrity (the file's si block, promote: an electrically long net is promoted to "
                    f"{CONTROLLED_CLASS} and routed again at its controlled width, as on a template's board):" if block.promote else
                    "Signal integrity (the file's si block, validation only: no net gains a routing rule, so the board is routed as without the block; "
                    "an electrically long net is reported by the SI checks, not routed as controlled impedance):")
    plan.changes.append(DesignChange(description="signal-integrity classes", target="si", operation="set", payload=si, rationale=note))


def _positive_number(raw: Any) -> bool:
    """Whether ``raw`` is a finite number > 0 (a bool is no number)."""
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return False
    try:
        return math.isfinite(float(raw)) and raw > 0
    except OverflowError:  # an integer too large for a float
        return False


def _check_sim_elements(design: DesignFile, build: _Builder) -> None:
    """Refuse ``simulation.nodes`` / ``simulation.elements`` the SPICE compiler would refuse (module docstring), naming the field.

    The compiler's own rules and judges, applied before anything is
    confirmed: its node-name rule, its element names
    (:func:`~ai_eda.compilers.spice.element_name`), the devices of
    :data:`~ai_eda.ir.simulation.SIM_ELEMENT_DEVICES`, the control pair of
    :data:`~ai_eda.ir.simulation.CONTROLLED_DEVICES`, its card judge and its
    ``SW`` card rule (:func:`~ai_eda.compilers.spice.switch_model_problem`),
    its parameter tokens, and its card registry: only the cards the netlist
    carries define a model (a part excluded from SPICE carries none), and a
    model name has one card. Runs after the parts are built, so a part's card
    was already judged.
    """
    sim = design.simulation
    if sim is None or not (sim.elements or sim.nodes):
        return
    nets = {n.name for n in design.nets}
    folded = {n.lower() for n in nets} | {"0", "gnd"}
    seen: set[str] = set()
    for i, node in enumerate(sim.nodes):
        where = f"--design: simulation.nodes.{i}"
        if not node or any(ch in _BAD_NODE_CHARS for ch in node) or not NODE_RE.fullmatch(node):
            raise DesignFileError(f"{where}: {node!r} is not a plain node name (letters, digits and _ . / + - : # @ [ ])")
        if node.lower() in folded:
            raise DesignFileError(f"{where}: {node!r} is a net of the file or the ground node (ngspice reads names case-insensitively); "
                                  "a simulation node is one no net has")
        if node.lower() in seen:
            raise DesignFileError(f"{where}: {node!r} is listed twice")
        seen.add(node.lower())
    known = nets | set(sim.nodes)
    refs = {p.ref.lower() for p in design.components}
    taken = {element_name(p.ref, p.spice.device).lower(): p.ref for p in design.components if p.spice.device is not None and not p.spice.exclude}
    # the cards the netlist carries, registered as the compiler registers them: model name (lower) -> its normalised card, and every
    # name a card defines (lower) -> (model | subckt, the model name whose card defines it)
    cards: dict[str, str] = {}
    defined: dict[str, tuple[str, str]] = {}

    def register(what: str, model_name: str, text: str, where: str) -> None:
        card, defs = check_model_card(what, model_name, text)
        key = model_name.lower()
        if key in cards and cards[key] != card:
            raise DesignFileError(f"{where}: {what}: model {model_name!r} has another, different model_card in the file; the netlist carries one "
                                  "card per model")
        if key in cards:
            return
        for dname in defs:
            if dname in defined and defined[dname][1] != key:
                raise DesignFileError(f"{where}: {what}: model_card defines {dname!r}, which the card for {defined[dname][1]!r} also defines")
        defined.update({dname: (dkind, key) for dname, (dkind, _ports) in defs.items()})
        cards[key] = card

    for j, p in enumerate(design.components):
        s = p.spice  # the compiler takes in the card of a part it netlists: not excluded, a device, no coupled line (it writes that card)
        if s.model_card is not None and s.model_name and not s.exclude and s.device is not None and s.device is not SpiceDevice.P:
            register(p.ref, s.model_name, s.model_card, f"--design: components.{j}.spice.model_card")
    for i, x in enumerate(sim.elements):
        where = f"--design: simulation.elements.{i}"
        who = f"element {x.id!r}"
        if not _NAME.fullmatch(x.id):  # fullmatch: the pattern's $ would let a trailing newline through into the netlist line
            raise DesignFileError(f"{where}.id: {x.id!r} must be a plain identifier (letters, digits, _): it becomes the element's netlist name")
        if x.id.lower() in refs:
            raise DesignFileError(f"{where}.id: {x.id!r} is a component ref of the file; a simulation element is no part - give it an id of its own")
        if x.device not in SIM_ELEMENT_DEVICES:
            raise DesignFileError(f"{where}.device: {who} is a {x.device.value}; a simulation-only element is R, C, L or D, or a voltage-controlled "
                                  "switch S (a source is a stimulus; a transistor or subcircuit is a part)")
        name = element_name(x.id, x.device)
        if name.lower() in taken:
            raise DesignFileError(f"{where}.id: {who} would be the netlist element {name}, which {taken[name.lower()]} already is (ngspice reads names case-insensitively)")
        taken[name.lower()] = f"simulation element {x.id}"
        for key in ("net", "reference_net"):
            if getattr(x, key) not in known:
                raise DesignFileError(f"{where}.{key}: {who} names {getattr(x, key)!r}, which is no net of the file and no simulation node (simulation.nodes)")
        if x.net == x.reference_net:
            raise DesignFileError(f"{where}.reference_net: {who}: net and reference_net are both {x.net!r}")
        controls = ("control_net", "control_reference_net")
        if x.device in CONTROLLED_DEVICES:
            for key in controls:
                if getattr(x, key) is None:
                    raise DesignFileError(f"{where}.{key}: {who} is a voltage-controlled switch: it needs both control nodes ({', '.join(controls)})")
                if getattr(x, key) not in known:
                    raise DesignFileError(f"{where}.{key}: {who} names {getattr(x, key)!r}, which is no net of the file and no simulation node (simulation.nodes)")
            if x.control_net == x.control_reference_net:
                raise DesignFileError(f"{where}.control_reference_net: {who}: control_net and control_reference_net are both {x.control_net!r} "
                                      "(the control voltage would be identically 0)")
            if x.params:
                raise DesignFileError(f"{where}.params: {who} is a voltage-controlled switch: it takes its numbers on its SW model card, no params "
                                      f"(got {sorted(x.params)})")
        else:
            given = [key for key in controls if getattr(x, key) is not None]
            if given:
                raise DesignFileError(f"{where}.{given[0]}: {who} is a {x.device.value}; only a voltage-controlled switch (S) has control nodes")
        if x.device in VALUE_DEVICES:
            stated = [key for key in ("model_name", "model_card", "model_source") if getattr(x, key) is not None]
            if stated:
                raise DesignFileError(f"{where}.{stated[0]}: {who} is a {x.device.value}: it takes a value, not a model")
            unit = DEVICE_UNITS[x.device]
            raw = x.value.value if isinstance(x.value, Quantity) else x.value
            if not _positive_number(raw):
                raise DesignFileError(f"{where}.value: {who} is a {x.device.value}: it needs a positive number in {unit}, got "
                                      f"{_plain(x.value) if x.value is not None else 'none'}")
            if isinstance(x.value, Quantity) and x.value.unit is not None and unit_key(x.value.unit) != unit_key(unit):
                raise DesignFileError(f"{where}.value: {who} is stated in {x.value.unit!r}, a {x.device.value} takes {unit}")
        else:
            if x.value is not None:
                raise DesignFileError(f"{where}.value: {who} is a {x.device.value}: it takes a model (model_name and its card), not a value")
            if not x.model_name or any(ch in _BAD_NODE_CHARS for ch in x.model_name):
                raise DesignFileError(f"{where}.model_name: {who} is a {x.device.value}: it needs a single-token model_name")
            if x.model_card is None and x.model_source is not None:
                raise DesignFileError(f"{where}.model_source: {who} has model_source without model_card")
            if x.model_card is not None:
                _element_card(x, where.removeprefix("--design: "))
                register(f"simulation element {x.id}", x.model_name, x.model_card, f"{where}.model_card")
        try:  # the compiler's parameter tokens: plain names, a number or a single SPICE token each
            _param_tokens(build.params(x.params), f"simulation element {x.id}", _Ledger())
        except CompileError as e:
            raise DesignFileError(f"{where}.params: {e}") from e
        except OverflowError as e:  # an integer too large for a float
            raise DesignFileError(f"{where}.params: simulation element {x.id}: a parameter is not a number ngspice can read ({e})") from e
        if not x.reason.strip() or len(x.reason.splitlines()) != 1 or not x.reason.strip().isprintable():
            raise DesignFileError(f"{where}.reason: {who} needs a reason: one line saying what it stands for and why it is not a part")
    for i, x in enumerate(sim.elements):  # every card is known now: a model may be defined by another element's card or a part's
        if x.device in VALUE_DEVICES:
            continue
        where = f"--design: simulation.elements.{i}." + ("model_card" if x.model_card is not None else "model_name")
        entry = defined.get(x.model_name.lower())  # type: ignore[union-attr]
        if entry is None:
            raise DesignFileError(f"{where}: no model_card in the file defines {x.model_name!r} (an element's, or a part's that is not excluded "
                                  "from SPICE: the netlist carries no other card and emits no .include)")
        kind, key = entry
        if kind != "model":
            raise DesignFileError(f"{where}: element {x.id!r} is a {x.device.value}: it needs a .model, but {x.model_name!r} is a .{kind}")
        if x.device in CONTROLLED_DEVICES:
            problem = switch_model_problem(cards[key], x.model_name)  # type: ignore[arg-type]
            if problem is not None:
                raise DesignFileError(f"{where}: element {x.id!r}: {problem}")


def _add_board(plan: DesignFilePlan, design: DesignFile, build: _Builder) -> None:
    """The file's ``board`` section into ``plan``'s layout (module docstring): checked, one table line each, one ``pcb`` change.

    The change sets a board holding only the layout (no stack, no classes);
    :meth:`DesignFilePlan.for_board` turns it into appends when the IR
    already has a board. A section that states nothing changes nothing.
    """
    b = design.board
    assert b is not None
    refs = {p.ref for p in design.components}
    if b.outline is not None:
        o = b.outline
        if not (o.width_mm > 0 and o.height_mm > 0):
            raise DesignFileError(f"--design: board.outline: width_mm and height_mm must be > 0, got {o.width_mm:g} x {o.height_mm:g}")
        plan.outline = BoardOutline(width_mm=o.width_mm, height_mm=o.height_mm, origin_x_mm=o.origin_x_mm, origin_y_mm=o.origin_y_mm)
    elif b.fixed or b.pack_regions or b.silk_texts:
        raise DesignFileError("--design: board.outline: the fixed parts, pack regions and silk texts are positions on the board's outline - state it "
                              "(width_mm, height_mm): the placement.fixed placer places only on a stated outline, and a text has no place without one")
    owner: dict[str, str] = {}  # ref -> where the section already put it
    for i, f in enumerate(b.fixed):
        where = f"--design: board.fixed.{i}.ref"
        if f.ref not in refs:
            raise DesignFileError(f"{where}: {f.ref!r} is no component of the file")
        if f.ref in owner:
            raise DesignFileError(f"{where}: {f.ref} is fixed twice")
        owner[f.ref] = "fixed"
        plan.fixed.append(Placement(component_ref=f.ref, x_mm=f.x_mm, y_mm=f.y_mm, rotation_deg=f.rotation_deg, side=f.side, provenance=build.provenance))
    ids: set[str] = set()
    for i, r in enumerate(b.pack_regions):
        where = f"--design: board.pack_regions.{i}"
        if r.id in ids:
            raise DesignFileError(f"{where}.id: pack region id {r.id!r} is used twice")
        ids.add(r.id)
        for j, ref in enumerate(r.refs):
            if ref not in refs:
                raise DesignFileError(f"{where}.refs.{j}: {ref!r} is no component of the file")
            if ref in owner:
                raise DesignFileError(f"{where}.refs.{j}: {ref} is already {owner[ref]}; a part is fixed or packed in one region, once")
            owner[ref] = f"in pack region {r.id}"
        try:
            region = PackRegion(id=r.id, rect=user_requirement(list(r.rect_mm), "mm", build.note), refs=list(r.refs), side=r.side, provenance=build.provenance)
        except ValidationError as e:
            raise DesignFileError(f"{where}: {e.errors()[0]['msg']}") from e
        if not _KEEPOUT_ID_RE.fullmatch(r.id):  # the IR's rule matches with $, which lets a trailing newline through into the table and the notes
            raise DesignFileError(f"{where}.id: pack region id {r.id!r} must be a plain identifier (a letter, then letters, digits or _)")
        x0, y0, x1, y1 = region.box()
        o = plan.outline
        assert o is not None  # a region needs the outline (above)
        if x0 < o.origin_x_mm or y0 < o.origin_y_mm or x1 > o.origin_x_mm + o.width_mm or y1 > o.origin_y_mm + o.height_mm:
            raise DesignFileError(f"{where}.rect_mm: pack region {r.id} ({x0:g}, {y0:g})-({x1:g}, {y1:g}) mm leaves the outline "
                                  f"({o.origin_x_mm:g}, {o.origin_y_mm:g})-({o.origin_x_mm + o.width_mm:g}, {o.origin_y_mm + o.height_mm:g}) mm")
        plan.pack_regions.append(region)
    sides = {p.ref: None for p in design.components}  # nothing is placed yet: the compiler's rule for a user / title text needs only the refs
    for i, t in enumerate(b.silk_texts):
        where = f"--design: board.silk_texts.{i}"
        if not t.text.strip() or not t.text.isascii() or not t.text.isprintable():
            raise DesignFileError(f"{where}.text: {t.text!r} must be one line of printable ASCII (KiCad's stroke font; the silkscreen placer's "
                                  "text-box estimate assumes Latin glyphs)")
        if t.ref is not None and t.ref not in refs:
            raise DesignFileError(f"{where}.ref: {t.ref!r} is no component of the file")
        o = plan.outline
        assert o is not None  # a text needs the outline (above)
        if not (o.origin_x_mm <= t.x_mm <= o.origin_x_mm + o.width_mm and o.origin_y_mm <= t.y_mm <= o.origin_y_mm + o.height_mm):
            raise DesignFileError(f"{where}: silk text {t.text!r} at ({t.x_mm:g}, {t.y_mm:g}) mm lies outside the outline "
                                  f"({o.origin_x_mm:g}, {o.origin_y_mm:g})-({o.origin_x_mm + o.width_mm:g}, {o.origin_y_mm + o.height_mm:g}) mm")
        text = SilkText(text=t.text, x_mm=t.x_mm, y_mm=t.y_mm, rotation_deg=t.rotation_deg, layer="B.SilkS" if t.side is BoardSide.BOTTOM else "F.SilkS",
                        size_mm=t.size_mm, thickness_mm=stroke_for(t.size_mm), justify=t.justify, kind=SilkKind(t.kind), component_ref=t.ref,
                        provenance=build.provenance)
        problem = silk_text_problems([text], sides).get(0)
        if problem is not None:
            raise DesignFileError(f"{where}: {problem}")
        plan.silk_texts.append(text)
    problem = design_file_layout_problem(plan)
    if problem is not None:
        raise DesignFileError(f"--design: board: {problem}")
    if not has_layout(plan):
        return
    plan.layout = layout_lines(plan)
    plan.changes.append(DesignChange(description="board layout", target="pcb", operation="set", payload=PCBDesign(**layout_fields(plan)),
                                     rationale=f"{build.note}; {_layout_summary(plan)}"))


def _build(design: DesignFile, sha256: str, library: KicadLibrary) -> DesignFilePlan:
    for what, names in (("component ref", [p.ref for p in design.components]), ("net name", [n.name for n in design.nets])):
        repeated = _duplicates(names)
        if repeated:
            raise DesignFileError(f"--design: {what}(s) {repeated} appear more than once")
    unannotated = [p.ref for p in design.components if not _REF_RE.fullmatch(p.ref)]
    if unannotated:  # the schematic compiler's rule, applied before anything is confirmed into the IR
        raise DesignFileError(f"--design: component ref(s) {unannotated} are not annotated (expected e.g. R1, J3)")
    untraced = [p.ref for p in design.components if not p.serves_requirements and not (p.spice.exclude and p.spice.exclude_reason.strip())]
    if untraced:
        raise DesignFileError(f"--design: part(s) {untraced} name no requirement: give each serves_requirements (the requirement ids it serves), or exclude it from SPICE with a reason if it has no electrical role")
    for p in design.components:
        # the templates' rule, for a circuit a person or a model writes: a part value is one that is sold, or names the real part that has it
        odd = _nonstandard_value(p)
        if odd is not None and "is stated in" in odd:
            raise DesignFileError(f"--design: {odd}")
        if odd is not None and not (p.value_source or "").strip():
            raise DesignFileError(f"--design: {odd}; use a standard value, or give the part a value_source (one line naming the real part that has this value)")
        if p.value_source is not None and (len(p.value_source.splitlines()) != 1 or not p.value_source.strip()):
            raise DesignFileError(f"--design: {p.ref} value_source must be one line of text naming the real part")
        layers = _min_layers_problem(p)
        if layers is not None:
            raise DesignFileError(f"--design: {layers}")
        identity = _identity_problem(p)
        if identity is not None:
            raise DesignFileError(f"--design: {identity}")
    build = _Builder(sha256, library)
    try:
        parts = [build.part(p) for p in design.components]
    except TemplateRefusal as e:
        raise DesignFileError(f"--design: {e}") from e
    by_ref = {c.ref: c for c in parts}
    nets = [build.net(n, by_ref) for n in design.nets]
    _check_sim_elements(design, build)
    t = design.topology
    topology = Topology(name=t.name, domains=t.domains, rationale=t.rationale, provenance=build.provenance,
                        schematic_layout=t.schematic_layout)
    topology.blocks = [Block(**b.model_dump(), provenance=build.provenance) for b in t.blocks]
    for a in t.pin_associations:
        def pin_ref(text: str) -> PinRef:
            ref, sep, number = text.rpartition(".")
            if not sep or not ref or not number:
                raise DesignFileError(f"--design: intent.pin_ref: {text!r} must be reference.pin")
            return PinRef(component_ref=ref, pin_number=number)
        topology.pin_associations.append(PinAssociation(
            terminal=pin_ref(a.terminal), ic_pin=pin_ref(a.ic_pin), role=a.role,
            provenance=Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note=f"{build.note}; {a.rationale}"),
        ))
    try:
        validate_intent(topology.blocks, topology.pin_associations, parts, nets)
    except ValueError as e:
        raise DesignFileError(f"--design: {e}") from e
    plan = DesignFilePlan(template=DESIGN_FILE_TEMPLATE, title=design.title, version=DESIGN_FILE_VERSION, sha256=sha256)
    plan.intent_lines = [f"block {b.id}: {b.function} ({b.domain.value}); members {', '.join(b.component_refs)}; inputs {', '.join(b.input_nets)}; outputs {', '.join(b.output_nets)}" for b in t.blocks]
    plan.intent_lines += [f"{a.terminal} -> {a.ic_pin}: {a.role}; {a.rationale}" for a in t.pin_associations]
    if t.schematic_layout is not None:
        plan.intent_lines.append(f"schematic layout: {t.schematic_layout}")
    plan.changes = [
        DesignChange(description=f"topology: {t.name}", target="topology", operation="set", payload=topology, rationale=build.note),
        *(DesignChange(description=f"add {c.ref} ({c.description})", target="components", operation="append", payload=c, rationale=build.note) for c in parts),
        *(DesignChange(description=f"add net {n.name}", target="nets", operation="append", payload=n, rationale=build.note) for n in nets),
    ]
    plan.parts = [f"{_part_line(c)}; {_spice_text(p.spice, p.ref)}{_part_extras(p)}" for c, p in zip(parts, design.components, strict=True)]
    plan.nets = [_net_line(n) + (f"; serves {', '.join(n.serves_requirements)}" if n.serves_requirements else "") for n in nets]
    plan.topology_line = f"{t.name} ({', '.join(d.value for d in t.domains)}){': ' + t.rationale if t.rationale else ''}"
    if design.simulation is not None:
        plan.changes.append(DesignChange(description="simulation setup", target="simulation", operation="set", payload=build.simulation(design.simulation), rationale=build.note))
        plan.simulation = _simulation_lines(design.simulation)
    if design.si is not None:
        _add_si(plan, design.si, build.note)
    if design.board is not None:
        _add_board(plan, design, build)
    return plan


def load_design_file(path: str | Path, library: KicadLibrary) -> DesignFilePlan:
    """The plan of the design file at ``path``; :class:`DesignFileError` with one line when it cannot be read or built."""
    try:
        raw = Path(path).read_bytes()
    except OSError as e:
        raise DesignFileError(f"--design: cannot read {path}: {e}") from e
    sha256 = "sha256:" + hashlib.sha256(raw).hexdigest()
    try:
        data = json.loads(raw.decode("utf-8-sig"), parse_constant=_no_constant)
    except (UnicodeDecodeError, ValueError) as e:
        raise DesignFileError(f"--design: {path} is not JSON: {e}") from e
    try:
        design = DesignFile.model_validate(data)
        return _build(design, sha256, library)
    except ValidationError as e:
        first = e.errors()[0]
        where = ".".join(str(x) for x in first["loc"]) or "(top level)"
        raise DesignFileError(f"--design: {path}: {where}: {first['msg']} ({e.error_count()} problem(s) in all)") from e


__all__ = ["DESIGN_FILE_FORMAT", "DESIGN_FILE_TEMPLATE", "DesignFileError", "DesignFilePlan", "NEEDS_EMPTY_IR", "load_design_file"]
