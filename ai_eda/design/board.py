"""Every template's board decisions: the layer stack (``pcb_layers`` 2 | 4) and the signal-integrity net classes (``ir.si``).

Invariant: nothing here is a fab fact or a datasheet fact. The stack is the
generic one of :mod:`ai_eda.design.stackup` ("a generic value, not a fab's")
and every SI number a template declares is a free choice shown in the
``confirm_design`` table and stamped like every other choice
(:func:`~ai_eda.design.base.choice_provenance`: ``assumption`` until the table
is confirmed, then ``user_requirement``), a value copied from a requirement,
or a registered calculator's output over those (``derived``, re-derived by
``calc.recompute``). SI treatment is need-driven:

* every template declares a ``DEFAULT`` class - no impedance target, the
  driver model the critical-length rule and the SPICE check use (edge
  :data:`T_RISE_S`, marked conservative; source resistance, far-end load,
  ringing band) - and the controlled-impedance class :data:`CONTROLLED_CLASS`
  (target :data:`Z0_TARGET_OHM` +/- :data:`Z0_TOL_REL`) with no nets: a net of
  ``DEFAULT`` is promoted into it only when its *routed* delay exceeds its
  critical length (:mod:`ai_eda.tools.si.promote`, the PCB agent). A board
  whose nets all stay short routes exactly as without classes;
* a template adds only what its circuit needs (:meth:`Template.si_declarations
  <ai_eda.design.base.Template.si_declarations>`): the ATmega128 board's
  crystal-loop length, the ISP timing path and the supply-rail minimum width.

The SI part (:func:`si_choices`, :func:`si_constraints`) is also what a
design file's optional ``si`` block gets (:mod:`ai_eda.design.design_file`):
the same ``DEFAULT`` class and table rows, any of the driver values and the
fraction stated by the user instead, and by default *no* controlled class and
no ``promote_to`` - validation only: no net gains a routing rule, so the
board routes exactly as without the block, and an electrically long net is
reported by the SI checks, not routed as controlled impedance.

The board layout part (:func:`layout_fields`, :func:`layout_onto`,
:func:`layout_lines`) is also what a design file's optional ``board``
section gets: the same ``PCBDesign`` fields, the same appending to a board the
IR already has (never replacing anything) and the same table lines - without
the stack and the SI classes :func:`add_board` adds for a template, so a file
that states a layout is routed and judged as one without it. Its silk texts
carry the design file's mark instead of a template's
(:func:`design_file_silk_text`), and the PCB agent keeps both kinds
(:func:`proposed_silk_text`).

The stack: ``pcb_layers`` (alias ``layer_count``) read by
:func:`~ai_eda.design.inputs.read_layer_count` under the template's
:attr:`~ai_eda.design.base.Template.layer_policy` (:func:`read_board_layers`);
without it the policy's default (2 layers unless the template states
another), shown as a choice. A stated count the policy does not allow refuses
the template with the policy's reason - before any missing input is asked
for (:func:`layer_policy_refusal`, called by the selection). A 4-layer stack
puts the ground plane on ``In1.Cu`` and the template's supply net
(:attr:`Template.plane_nets`) on ``In2.Cu``, and adds the plane edge
clearance as the parameter :data:`PLANE_CLEARANCE_KEY` (the PCB agent draws
the plane zones from it). An IR that already carries a stack (a fab's, the
user's) keeps it: the template proposes no stack then and says so - unless
the template's policy restricts the counts and that stack's count is not one
of them (a 2-layer stack under a template whose lines need a plane), which
refuses the template. The generic policy never refuses an existing stack, so
every template without its own policy behaves exactly as before.

The base material (:mod:`ai_eda.design.materials`,
:func:`~ai_eda.design.materials.select_material`): a board whose requirements
name no material and no material need is built exactly as above (FR-4, byte
for byte). Otherwise the stack is the decided material's - the user's
``pcb_material``, or the rule's proposal shown as a ``pcb_material`` row of
the table - with its layer count (:func:`~ai_eda.design.materials.material_layer_count`:
flex 2, IMS 1 copper layer, the high-frequency laminate 2 or 4), its rows and
board lines, and its provenance naming the requirements it was built from (so
it serves them). A material the rule cannot decide (a power-dissipation need,
two needs proposing different materials) is the required question
``pcb_material`` and no design; an unusable or contradicting statement
refuses with the reason; a material stated beside an IR's own stack refuses
(that stack is kept, the material could not be applied to it).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from ai_eda.ir import (
    CircuitIR,
    Keepout,
    MissingInformation,
    NetClass,
    PCBDesign,
    Provenance,
    ProvenanceKind,
    SIConstraints,
    SilkKind,
    SilkText,
    Stackup,
    TimingPath,
    Traced,
)

from ai_eda.design.base import TOOL_ID as TEMPLATE_TOOL_ID
from ai_eda.design.base import (
    DESIGN_FILE_NOTE,
    Choice,
    DesignChange,
    Plan,
    choice_provenance,
    leave_out_answer,
    leave_out_remedy,
    refusal_question,
    structural_provenance,
    template_tool,
    unusable_remedy,
)
from ai_eda.design.inputs import LAYER_COUNT_ALIASES, LAYER_COUNT_KEY, MATERIAL_KEY, DesignInput, LayerCountInput, read_layer_count
from ai_eda.design.materials import (
    DEFAULT_MATERIAL,
    MATERIALS,
    material_board_lines,
    material_layer_count,
    material_rows,
    material_stack,
    material_stackup,
    select_material,
)
from ai_eda.design.stackup import PLANE_EDGE_CLEARANCE_MM, board_layers, generic_stackup, stackup_choices

if TYPE_CHECKING:
    from ai_eda.design.base import Template

#: the conservative driver edge every template's classes assume unless a datasheet t_rise is grounded (s)
T_RISE_S = 1e-9
#: a typical CMOS output's source resistance at 5 V (ohm) and a typical CMOS input with its pad (F): the SPICE check's driver / load
R_DRIVE_OHM = 25.0
C_LOAD_F = 5e-12
#: overshoot / undershoot / settling band of the SPICE check, as a fraction of the swing
RINGING_TOL_REL = 0.15
#: the critical-length rule's fraction of the rise time (1/2: the round trip is shorter than the edge)
CRITICAL_FRACTION = 0.5
#: the controlled-impedance class a long net is promoted to
CONTROLLED_CLASS = "Z50"
Z0_TARGET_OHM = 50.0
Z0_TOL_REL = 0.10
#: the default class's name
DEFAULT_CLASS = "DEFAULT"
#: the parameter holding the 4-layer plane zones' inset from the board edge (mm)
PLANE_CLEARANCE_KEY = "plane_edge_clearance"


@dataclass
class BoardContext:
    """What :meth:`Template.si_declarations` may use: the plan's parameters, the stack, the shared driver choices, and a way to add choices."""

    template_id: str
    confirmed: bool
    params: dict[str, Traced]
    stackup: Stackup | None
    driver: dict[str, Traced]
    choices: list[Choice] = field(default_factory=list)
    new_params: dict[str, Traced] = field(default_factory=dict)
    computed: list[tuple[str, Traced]] = field(default_factory=list)

    def choice(self, key: str, value: Any, unit: str | None, description: str, *, param: bool = False) -> Traced:
        """A free choice shown in the table (``param``: also an ``ir.parameters`` entry, for a calculator to read)."""
        self.choices.append(Choice(key, description, value, unit))
        t = Traced(value=value, unit=unit, provenance=choice_provenance(self.template_id, f"{key} = {value!r}{' ' + unit if unit else ''}: {description}", self.confirmed))
        if param:
            self.new_params[key] = t
        return t

    def computed_param(self, key: str, traced: Traced) -> Traced:
        self.new_params[key] = traced
        self.computed.append((key, traced))
        return traced

    def stated(self, key: str, traced: Traced, choice: Any, unit: str | None, description: str) -> Traced:
        """The user's own value in place of the choice ``key`` (``choice`` / ``unit``: what it replaces): a table row that says so, kept as given."""
        self.choices.append(Choice(key, f"your value, stated in place of the choice {choice:g}{' ' + unit if unit else ''} ({description})", traced.value, traced.unit))
        return traced

    def structural(self, note: str) -> Provenance:
        return structural_provenance(self.template_id, note)


@dataclass
class SIDeclarations:
    """What a template declares beyond the default classes: its own classes and timing paths, and who drives the default nets."""

    classes: list[NetClass] = field(default_factory=list)
    timing_paths: list[TimingPath] = field(default_factory=list)
    #: the component whose grounded datasheet facts replace the driver choices of the DEFAULT / controlled classes
    driver: str | None = None
    #: one line per class / path for the confirmation table
    lines: list[str] = field(default_factory=list)


def _plan_params(plan: Plan) -> dict[str, Traced]:
    return {c.target.removeprefix("parameters."): c.payload for c in plan.changes if c.target.startswith("parameters.") and isinstance(c.payload, Traced)}


def _fmt(t: Traced | None, unit: str = "") -> str:
    if t is None:
        return "-"
    return f"{float(t.value):.6g}{' ' + unit if unit else ''}"


def class_line(c: NetClass) -> str:
    """One table line for a class: its nets and every constraint it states."""
    parts = [f"{c.name}{' (default: every net no class lists)' if c.default else ''}: nets {', '.join(c.nets) if c.nets else '-'}"]
    if c.target_z0_ohm is not None:
        parts.append(f"Z0 {_fmt(c.target_z0_ohm, 'ohm')} +/- {float(c.z0_tol_rel.value):.0%}")  # type: ignore[union-attr]
    if c.max_length_mm is not None:
        parts.append(f"max length {_fmt(c.max_length_mm, 'mm')}")
    if c.max_delay_s is not None:
        parts.append(f"max delay {_fmt(c.max_delay_s, 's')}")
    if c.min_width_mm is not None:
        parts.append(f"min width {float(c.min_width_mm.value):.6g} mm [{c.min_width_mm.provenance.tool or c.min_width_mm.provenance.kind.value}]")
    if c.t_rise_s is not None:
        parts.append(f"driver t_r {_fmt(c.t_rise_s, 's')}, R {_fmt(c.r_drive_ohm, 'ohm')}, load {_fmt(c.c_load_f, 'F')}"
                     + (f" (grounded {c.driver} facts replace them on the nets {c.driver} drives)" if c.driver else " (no driver named: these values stand)"))
    if c.promote_to is not None:
        parts.append(f"a net longer than its critical length is promoted to {c.promote_to}")
    if c.xtalk_budget_rel is not None:  # only the user's stated budget exists (NetClass refuses any other): no template class has one
        parts.append(f"crosstalk budget {float(c.xtalk_budget_rel.value):.6g} of the swing (yours: it takes precedence over the NM / 3 budget "
                     "spice.xtalk computes from grounded datasheet levels)")
    if c.description and not c.default and c.target_z0_ohm is None:
        parts.append(c.description)  # a template class says in words why it is there (and, without an edge, why no rule applies)
    return "; ".join(parts)


def keepout_line(k: Keepout) -> str:
    """One table line for a keep-out the template proposes: its area, layers, bans, exceptions and reason."""
    area = (f"rect x {k.rect.value[0]:g} y {k.rect.value[1]:g} w {k.rect.value[2]:g} h {k.rect.value[3]:g} mm" if k.rect is not None
            else f"polygon of {len(k.polygon.value)} points (mm)")  # type: ignore[union-attr]
    extra = "".join(f"; allowed {label} {', '.join(names)}" for label, names in (("refs", k.allowed_refs), ("nets", k.allowed_nets)) if names)
    return f"keep-out {k.id}: {area} on {', '.join(k.layers)}, no {', '.join(k.forbids)}{extra} - {k.reason}"


def template_silk_text(t: SilkText) -> bool:
    """Whether ``t`` is a board text a template put on the silk: a ``user`` / ``title`` text with the structural provenance of a template
    (``design.template.<id>``, :func:`~ai_eda.design.base.structural_provenance`).

    The PCB agent keeps such texts where they are and places the silkscreen
    around them; any other text in ``ir.pcb.silkscreen`` (a reference, a pin
    label, a text the user drew) still means the board's silkscreen is
    designed and is never replaced - except a design file's board text
    (:func:`design_file_silk_text`, :func:`proposed_silk_text`).
    :func:`add_board` refuses a plan text that would not pass this test, so a
    template's texts are always kept.
    """
    return t.kind in (SilkKind.USER, SilkKind.TITLE) and bool(t.provenance.tool) and t.provenance.tool.startswith(f"{TEMPLATE_TOOL_ID}.")  # type: ignore[union-attr]


#: the provenance note of every value a design file put in the IR: ``design file sha256:<the file's hash>`` (:data:`~ai_eda.design.base.DESIGN_FILE_NOTE`)
_DESIGN_FILE_MARK = re.compile(re.escape(DESIGN_FILE_NOTE) + r"sha256:[0-9a-f]{64}")


def design_file_silk_text(t: SilkText) -> bool:
    """Whether ``t`` is a board text a design file put on the silk (its ``board.silk_texts``): a ``user`` / ``title`` text stamped as
    every value of a design file is - ``user_requirement``, no tool, the note ``design file sha256:<the file's hash>`` and nothing else.

    A design file states no provenance, so its texts cannot carry a
    template's structural mark (and are never given one: the file is no
    template). They reach ``ir.pcb.silkscreen`` only through the
    ``confirm_design`` table, as a template's do, so the PCB agent keeps them
    the same way (:func:`proposed_silk_text`). A text the user drew into the
    IR by hand carries no such note and still stops the silkscreen placer.
    """
    p = t.provenance
    return (t.kind in (SilkKind.USER, SilkKind.TITLE) and p.kind is ProvenanceKind.USER_REQUIREMENT and not p.tool
            and _DESIGN_FILE_MARK.fullmatch(p.note or "") is not None)


def proposed_silk_text(t: SilkText) -> bool:
    """Whether ``t`` is a board text proposed with the design and confirmed with it: a template's (:func:`template_silk_text`) or a design
    file's (:func:`design_file_silk_text`). The PCB agent keeps such texts where they are and places the silkscreen around them."""
    return template_silk_text(t) or design_file_silk_text(t)


def has_layout(plan: Plan) -> bool:
    """Whether ``plan`` proposes a board layout (:attr:`~ai_eda.design.base.Plan.outline` / ``fixed`` / ``pack_regions`` / ``silk_texts``)."""
    return plan.outline is not None or bool(plan.fixed or plan.pack_regions or plan.silk_texts)


def layout_lines(plan: Plan) -> list[str]:
    """One table line for the outline, each fixed part, each pack region and each silk text the template proposes (none without them)."""
    out: list[str] = []
    if plan.outline is not None:
        o = plan.outline
        out.append(f"board outline: {o.width_mm:g} x {o.height_mm:g} mm at ({o.origin_x_mm:g}, {o.origin_y_mm:g}) mm")
    out += [f"fixed part {p.component_ref}: at ({p.x_mm:g}, {p.y_mm:g}) mm, rotation {p.rotation_deg:g} deg, {p.side} side" for p in plan.fixed]
    for r in plan.pack_regions:
        x, y, w, h = (float(v) for v in r.rect.value)
        out.append(f"pack region {r.id}: rect x {x:g} y {y:g} w {w:g} h {h:g} mm, {r.side} side, packed in this order: {', '.join(r.refs)}")
    out += [f"silk text {t.text!r} ({t.kind}): at ({t.x_mm:g}, {t.y_mm:g}) mm on {t.layer}, {t.size_mm:g} mm"
            + (f", beside {t.component_ref}" if t.component_ref else "") for t in plan.silk_texts]
    return out


def _layout_summary(plan: Plan) -> str:
    """``board layout: outline, 8 fixed part(s), ...`` - what a plan's layout holds, for a change's rationale."""
    held = (["outline"] if plan.outline is not None else []) + [f"{len(items)} {what}" for what, items in (
        ("fixed part(s)", plan.fixed), ("pack region(s)", plan.pack_regions), ("silk text(s)", plan.silk_texts)) if items]
    return "board layout: " + ", ".join(held)


def _layout_problem(template_id: str, plan: Plan) -> str | None:
    """Why the plan's layout cannot be proposed (a text another template's or nobody's, a layout the IR model refuses), or ``None``."""
    own = template_tool(template_id)
    for t in plan.silk_texts:
        if not template_silk_text(t) or t.provenance.tool != own:
            return (f"silk text {t.text!r} ({t.kind}, provenance tool {t.provenance.tool!r}): a template's board text is a user / title text with "
                    f"its structural provenance ({own}), the mark by which the PCB agent keeps it")
    return _layout_model_problem(plan, "the template's board layout")


def _layout_model_problem(plan: Plan, what: str) -> str | None:
    """Why the IR model refuses the plan's layout (a part fixed twice, a part both fixed and packed, ...), or ``None``."""
    try:
        PCBDesign(outline=plan.outline, fixed=list(plan.fixed), pack_regions=list(plan.pack_regions), silkscreen=list(plan.silk_texts))
    except ValidationError as e:
        return f"{what} is refused: " + "; ".join(str(err.get("msg", "")) for err in e.errors())
    return None


def design_file_layout_problem(plan: Plan) -> str | None:
    """Why a design file's board section cannot be proposed as its plan's layout, or ``None``: the template rule of :func:`add_board`
    with the design file's own mark - every text a :func:`design_file_silk_text` (the mark by which the PCB agent keeps it), and a
    layout the IR model accepts."""
    for t in plan.silk_texts:
        if not design_file_silk_text(t):
            return (f"silk text {t.text!r} ({t.kind}): a design file's board text is a user / title text stamped as every value of the file "
                    "(user_requirement, the note 'design file sha256:...'), the mark by which the PCB agent keeps it")
    return _layout_model_problem(plan, "the file's board section")


def layout_fields(plan: Plan) -> dict[str, Any]:
    """The :class:`~ai_eda.ir.PCBDesign` fields of the plan's board layout - only those it states (``{}`` for a plan without one)."""
    extra: dict[str, Any] = {}
    if plan.outline is not None:
        extra["outline"] = plan.outline
    for key, items in (("fixed", plan.fixed), ("pack_regions", plan.pack_regions), ("silkscreen", plan.silk_texts)):
        if items:
            extra[key] = list(items)
    return extra


def _layout_clash(pcb: PCBDesign, plan: Plan, who: str = "the template") -> str | None:
    """Why the plan's layout cannot be appended to the existing ``pcb`` (it never replaces anything there), or ``None``."""
    if plan.outline is not None and pcb.outline is not None:
        return f"ir.pcb already has an outline ({pcb.outline.width_mm:g} x {pcb.outline.height_mm:g} mm); {who} never replaces it"
    if (plan.fixed or plan.pack_regions) and pcb.placements:
        return (f"ir.pcb already has {len(pcb.placements)} placement(s): the PCB agent never re-places a board, so {who}'s fixed parts "
                "and pack regions would never be read")
    taken = {p.component_ref: "fixed" for p in pcb.fixed} | {ref: f"in pack region {r.id}" for r in pcb.pack_regions for ref in r.refs}
    clash = [f"{ref} ({taken[ref]})" for ref in [*(p.component_ref for p in plan.fixed), *(ref for r in plan.pack_regions for ref in r.refs)] if ref in taken]
    if clash:
        return f"part(s) {', '.join(clash)} already in ir.pcb; {who} never replaces a fixed part or a pack region"
    have_ids = {r.id for r in pcb.pack_regions}
    ids = [r.id for r in plan.pack_regions if r.id in have_ids]
    if ids:
        return f"pack region(s) {ids} already exist in ir.pcb.pack_regions; {who} never replaces one"
    have = {(t.kind, t.text, t.layer) for t in pcb.silkscreen}
    dup = [t.text for t in plan.silk_texts if (t.kind, t.text, t.layer) in have]
    if dup:
        return f"silk text(s) {dup} already exist in ir.pcb.silkscreen; {who} never doubles a text"
    return None


def layout_onto(pcb: PCBDesign, plan: Plan, rationale: str, who: str = "the template") -> tuple[list[DesignChange], str | None]:
    """``(changes, None)`` appending the plan's board layout to the existing board ``pcb`` (the outline set, each fixed part, pack region
    and silk text appended - never replacing anything there), or ``([], why)`` when it clashes with what ``pcb`` holds."""
    clash_why = _layout_clash(pcb, plan, who)
    if clash_why is not None:
        return [], clash_why
    changes: list[DesignChange] = []
    if plan.outline is not None:
        changes.append(DesignChange(description="board outline", target="pcb.outline", operation="set", payload=plan.outline, rationale=rationale))
    changes += [DesignChange(description=f"fixed part {p.component_ref}", target="pcb.fixed", operation="append", payload=p, rationale=rationale)
                for p in plan.fixed]
    changes += [DesignChange(description=f"pack region {r.id}", target="pcb.pack_regions", operation="append", payload=r, rationale=rationale)
                for r in plan.pack_regions]
    changes += [DesignChange(description=f"silk text {x.text!r}", target="pcb.silkscreen", operation="append", payload=x, rationale=rationale)
                for x in plan.silk_texts]
    return changes, None


def path_line(p: TimingPath) -> str:
    known = [k for k in ("t_co_max_s", "t_co_min_s", "t_su_min_s", "t_h_min_s") if getattr(p, k) is not None]
    return (
        f"timing path {p.name}: clock {p.clock_net} -> data {', '.join(p.data_nets)} ({p.direction}); f_clk {_fmt(p.f_clk_hz, 'Hz')}, "
        f"capture after {_fmt(p.capture_fraction)} of the period; datasheet terms " + (", ".join(f"{k} <- {v}" for k, v in sorted(p.terms_from.items())) or "-")
        + (f"; given: {', '.join(known)}" if known else "; none grounded yet (setup / hold margins stay NOT_VERIFIED until they are)")
    )


def _not_allowed(template: Template, layers: LayerCountInput) -> str:
    """Why a stated layer count refuses ``template`` (its policy does not build it)."""
    policy = template.layer_policy
    rid = layers.requirement.id if layers.requirement is not None else LAYER_COUNT_KEY
    return (f"{rid}: {layers.value} layers is not a count template {template.id} builds (it builds {policy.allowed_text()} layers)"
            + (f": {policy.reason.strip()}" if policy.reason.strip() else ""))


def read_board_layers(template: Template, ir: CircuitIR) -> tuple[LayerCountInput | None, str | None]:
    """``(layers, None)`` with the board's layer count under ``template``'s layer policy, or ``(None, why)``.

    :func:`~ai_eda.design.inputs.read_layer_count` reads the stated count (an
    unreadable, ambiguous or non-generic one is refused there); no stated
    count gives the policy's default (still ``is_default``: a choice the table
    shows); a stated count the policy does not allow is refused with the
    policy's reason. Under the generic policy this is ``read_layer_count``.
    """
    layers, why = read_layer_count(ir)
    if layers is None:
        return None, why
    policy = template.layer_policy
    if layers.is_default:
        return (layers if layers.value == policy.default else LayerCountInput(value=policy.default)), None
    if layers.value not in policy.allowed:
        return None, _not_allowed(template, layers)
    return layers, None


def layer_policy_refusal(template: Template, ir: CircuitIR) -> MissingInformation | None:
    """A non-required question under the requirement's own key when a stated, readable count is outside ``template``'s policy; else ``None``.

    The selection calls it before it asks for any missing input: an input the
    user already stated that refuses the template makes every answer to a
    required question useless. The requirement exists, so the question asks
    for the requirement to change (or another design), never for an
    ``--answer`` under its key (a typed value is kept) - and names the answer
    that leaves it out (``--answer leave_out=<key>``: the template's default
    stack is then used).
    """
    layers, _why = read_layer_count(ir)
    if layers is None or layers.is_default or layers.requirement is None or layers.value in template.layer_policy.allowed:
        return None
    why = _not_allowed(template, layers)
    key = layers.requirement.key
    return refusal_question(
        key,
        (f"{why}; no template design was proposed. Change the requirement to {template.layer_policy.allowed_text()} layers "
         f"(or leave it out: the template's default is {template.layer_policy.default} layers), or choose another design. "
         f"{leave_out_remedy(key, where='this design')}."),
        why,
    )


#: what a design file's ``si`` block reports instead of a promotion (``promote`` false: validation only)
VALIDATION_ONLY = ("is reported by the SI checks (si.critical_length, spice.si), not routed as controlled impedance: validation only, "
                   "nothing is promoted and no net gains a routing rule")


def si_choices(ctx: BoardContext, *, promote: bool = True, stated: dict[str, Traced] | None = None) -> tuple[Traced, Traced | None, Traced | None]:
    """The driver model of the ``DEFAULT`` class (``ctx.driver``) and ``(critical fraction, Z0 target, Z0 tolerance)``, each a table row.

    Every number is a free choice (:meth:`BoardContext.choice`) unless
    ``stated`` holds the user's own value under its key (``si.t_rise``,
    ``si.r_drive``, ``si.c_load``, ``si.ringing_tol``, ``si.critical_fraction``:
    a design file's ``si`` block), which is kept with its own provenance and
    shown as the user's (:meth:`BoardContext.stated`). Without ``promote``
    there is no controlled class: no Z0 choice is made (``None``, ``None``)
    and the fraction's row says that a net longer than ``l_crit`` is
    reported by the SI checks, not routed as controlled impedance.
    """
    stated = stated or {}

    def pick(key: str, value: float, unit: str | None, description: str) -> Traced:
        given = stated.get(key)
        return ctx.choice(key, value, unit, description) if given is None else ctx.stated(key, given, value, unit, description)

    ctx.driver = {
        "t_rise_s": pick("si.t_rise", T_RISE_S, "s", (
            "driver edge for the critical-length rule and the SPICE check: 1 ns, conservative (faster than the parts are expected to switch; "
            "not a measured value - a grounded datasheet t_rise of the driver replaces it)")),
        "r_drive_ohm": pick("si.r_drive", R_DRIVE_OHM, "ohm", "driver source resistance of the SPICE check (a typical CMOS output at 5 V; not a datasheet value)"),
        "c_load_f": pick("si.c_load", C_LOAD_F, "F", "far-end load of the SPICE check (a typical CMOS input with its pad; not a datasheet value)"),
        "ringing_tol_rel": pick("si.ringing_tol", RINGING_TOL_REL, None, "overshoot / undershoot / settling band of the SPICE check (15 % of the swing)"),
    }
    fraction = pick("si.critical_fraction", CRITICAL_FRACTION, None, (
        "critical-length rule l_crit = fraction x t_r / t_pd: 1/2 (the round trip is shorter than the edge); a routed net longer than l_crit "
        + (f"is promoted to {CONTROLLED_CLASS}" if promote else VALIDATION_ONLY)))
    if not promote:
        return fraction, None, None
    z0 = ctx.choice("si.z0_target", Z0_TARGET_OHM, "ohm", f"target impedance of the controlled class {CONTROLLED_CLASS} (defined only over a reference plane)")
    z0_tol = ctx.choice("si.z0_tol", Z0_TOL_REL, None, f"tolerance of the {CONTROLLED_CLASS} impedance (10 %)")
    return fraction, z0, z0_tol


def si_constraints(ctx: BoardContext, decl: SIDeclarations, fraction: Traced, z0: Traced | None, z0_tol: Traced | None, *,
                   xtalk_budget: Traced | None = None) -> SIConstraints:
    """``ir.si``: the ``DEFAULT`` class with ``ctx.driver``, the controlled class when :func:`si_choices` made one (``z0``), then ``decl``'s.

    ``xtalk_budget`` (the user's stated crosstalk budget, a fraction of the
    swing) goes on the default and the controlled class; ``None`` leaves
    both without one (spice.xtalk then computes NM / 3 from grounded levels).
    """
    drv = ctx.driver
    promote = z0 is not None
    default = NetClass(
        name=DEFAULT_CLASS, default=True,
        description=("every net no other class lists: routed at the board rules unless the critical-length rule promotes it" if promote
                     else "every net no other class lists: routed at the board rules; an electrically long net " + VALIDATION_ONLY),
        t_rise_s=drv["t_rise_s"], r_drive_ohm=drv["r_drive_ohm"], c_load_f=drv["c_load_f"], ringing_tol_rel=drv["ringing_tol_rel"], driver=decl.driver,
        promote_to=CONTROLLED_CLASS if promote else None, xtalk_budget_rel=xtalk_budget, provenance=ctx.structural("default net class"),
    )
    classes = [default]
    if promote:
        classes.append(NetClass(
            name=CONTROLLED_CLASS, description="controlled impedance for nets the critical-length rule finds electrically long",
            target_z0_ohm=z0, z0_tol_rel=z0_tol, t_rise_s=drv["t_rise_s"], r_drive_ohm=drv["r_drive_ohm"], c_load_f=drv["c_load_f"],
            ringing_tol_rel=drv["ringing_tol_rel"], driver=decl.driver, xtalk_budget_rel=xtalk_budget,
            provenance=ctx.structural("controlled-impedance class for promoted nets"),
        ))
    note = ("signal-integrity classes: need-driven (declared interfaces + the critical-length rule)" if promote
            else "signal-integrity classes: validation only (the SI checks judge the routed copper; no net is promoted)")
    return SIConstraints(net_classes=[*classes, *decl.classes], timing_paths=list(decl.timing_paths), critical_fraction=fraction,
                         provenance=ctx.structural(note))


def add_board(template: Template, ir: CircuitIR, plan: Plan, *, confirmed: bool) -> str | None:
    """Append the board's stack and SI classes to a buildable ``plan`` (choices, inputs, table lines, changes); the refusal reason otherwise.

    The plan's own keep-outs (:attr:`~ai_eda.design.base.Plan.keepouts`, the
    transceiver's antenna band) go into the proposed ``PCBDesign`` with the
    stack, or are appended to an existing ``ir.pcb`` (one whose id is already
    there refuses: a keep-out is never replaced); each gets a table line. A
    plan without keep-outs proposes exactly the board it proposed before.

    The plan's board layout (:attr:`~ai_eda.design.base.Plan.outline`,
    ``fixed``, ``pack_regions``, ``silk_texts``: a keyboard's key row, its
    note names) goes the same way: into the proposed ``PCBDesign``, or
    appended to an existing ``ir.pcb`` - refusing an outline where one is
    already set, a ref already fixed or packed there, a region id already
    there, a text already there, and fixed parts / regions on a board that is
    already placed (the PCB agent would never read them); each item gets a
    table line (:func:`layout_lines`). A silk text that is not a ``user`` /
    ``title`` text stamped with this template's structural provenance
    (:func:`template_silk_text`) and a layout the IR model refuses (a part
    fixed twice, a part both fixed and packed) refuse the plan. A plan
    without a layout proposes exactly the board it proposed before.
    """
    t = template.id
    policy = template.layer_policy
    layers, why = read_board_layers(template, ir)
    if layers is None:
        # a stated count no reader can use (or one the policy refuses) is cleared by leaving it out: a typed value is kept
        remedy = unusable_remedy(ir, LAYER_COUNT_ALIASES)
        return f"{LAYER_COUNT_KEY}: {why}" + (f"; {remedy}" if remedy else "")
    layout = has_layout(plan)
    if layout:
        problem = _layout_problem(t, plan)
        if problem is not None:
            return problem
    params = _plan_params(plan)
    board_params: dict[str, Traced] = {}
    changes: list[DesignChange] = []
    existing = ir.pcb.stackup if ir.pcb is not None else None
    ground, power = template.plane_nets
    power = power or ground
    material = select_material(ir)  # untouched (nothing about the material stated or needed): the generic FR-4 path below, byte for byte
    if material.refusal is not None:
        return material.refusal
    if material.question is not None:  # the rule asks instead of deciding: no design until the user answers pcb_material
        plan.questions.append(material.question)
        return material.question.rationale
    stack: Stackup | None
    if existing is not None:
        if policy.restricts and existing.layer_count not in policy.allowed:
            return (f"{LAYER_COUNT_KEY}: the IR's own {existing.layer_count}-layer stack is not a count template {t} builds "
                    f"(it builds {policy.allowed_text()} layers): {policy.reason.strip()}")
        if not material.untouched:
            keys = list(dict.fromkeys(r.key for r in material.read))
            return (f"{MATERIAL_KEY}: the IR already carries its own {existing.layer_count}-layer stack, which the template keeps, so the base material "
                    f"({MATERIALS[material.material].name}: {material.reason}) is not applied to it; remove that stack or leave the material keys out "
                    f"({leave_out_answer(keys)})")
        stack = existing
        plan.board.append(f"stackup: the IR's own {existing.layer_count}-layer stack is kept (no generic stack proposed)")
    else:
        count, why = material_layer_count(material, layers, policy)
        if count is None:
            return why
        ms = None if material.material == DEFAULT_MATERIAL else material_stack(material.material, count)
        if ms is None:
            plan.choices.extend(stackup_choices(layers, ground_net=ground, power_net=power, policy=policy))
            stack = generic_stackup(layers.value, t, confirmed=confirmed, ground_net=ground, power_net=power)
        else:
            plan.choices.extend(material_rows(ms, material, layers, policy, ground_net=ground, power_net=power))
            stack = material_stackup(ms, material, t, confirmed=confirmed, ground_net=ground, power_net=power)
        served: list[str] = []
        if layers.traced is not None:  # the count is the user's: the stack's provenance says which requirement it serves
            inp = DesignInput(key=LAYER_COUNT_KEY, requirement=layers.requirement, traced=layers.traced)  # type: ignore[arg-type]
            plan.inputs[LAYER_COUNT_KEY] = inp
            board_params[LAYER_COUNT_KEY] = layers.traced
            served.append(layers.requirement.id)  # type: ignore[union-attr]
        served += [r.id for r in material.read if r.id not in served]  # and the material requirements it was built from
        if served:
            prov = stack.provenance.model_copy(update={"derived_from": served})
            stack = stack.model_copy(update={"provenance": prov})
        if stack.plane_layers():
            board_params[PLANE_CLEARANCE_KEY] = Traced(
                value=PLANE_EDGE_CLEARANCE_MM, unit="mm",
                provenance=choice_provenance(t, f"stackup.plane_edge_clearance = {PLANE_EDGE_CLEARANCE_MM!r} mm: plane zones end this far inside the board edge", confirmed),
            )
        n = layers.value if ms is None else ms.layer_count
        what = f"generic {n}-layer stack" if ms is None else ms.title
        if ir.pcb is None:
            extra = layout_fields(plan)  # the layout only when the plan has one: every other board is the PCBDesign it was before
            changes.append(DesignChange(description=f"board stack: {n} layers", target="pcb", operation="set",
                                        payload=PCBDesign(layers=board_layers(stack), stackup=stack, keepouts=list(plan.keepouts), **extra),
                                        rationale=f"template {t}: {what}" + (f" and {len(plan.keepouts)} keep-out(s)" if plan.keepouts else "")
                                        + (f"; {_layout_summary(plan)}" if layout else "")))
        else:
            changes.append(DesignChange(description=f"board layers: {n}", target="pcb.layers", operation="set", payload=board_layers(stack),
                                        rationale=f"template {t}: {what}"))
            changes.append(DesignChange(description=f"board stack: {n} layers", target="pcb.stackup", operation="set", payload=stack,
                                        rationale=f"template {t}: {what}"))
        source = f"from {layers.requirement.id}" if layers.requirement is not None else "the default"
        if ms is None:
            plan.board.append(f"stackup: generic {layers.value}-layer ({source}); "
                              + ("planes In1.Cu = " + ground + ", In2.Cu = " + power if layers.value == 4 else "no plane: impedance is undefined, only delay bounds are computed"))
            if not material.untouched:
                plan.board.extend([f"base material: {MATERIALS[DEFAULT_MATERIAL].name} ({material.reason})", *material.notes])
        else:
            plan.board.extend(material_board_lines(ms, material, source if ms.layer_count == layers.value else f"{ms.layer_count} copper layer(s): the material's stack",
                                                   ground, power))
    if ir.pcb is not None and plan.keepouts:
        have = {k.id for k in ir.pcb.keepouts}
        clash = sorted(k.id for k in plan.keepouts if k.id in have)
        if clash:
            return f"keep-out(s) {clash} already exist in ir.pcb.keepouts; the template never replaces a keep-out"
        changes += [DesignChange(description=f"keep-out {k.id}", target="pcb.keepouts", operation="append", payload=k, rationale=f"template {t}: {k.reason}")
                    for k in plan.keepouts]
    plan.board.extend(keepout_line(k) for k in plan.keepouts)
    if ir.pcb is not None and layout:
        appended, clash_why = layout_onto(ir.pcb, plan, f"template {t}: {_layout_summary(plan)}")
        if clash_why is not None:
            return clash_why
        changes += appended
    plan.board.extend(layout_lines(plan))
    ctx = BoardContext(template_id=t, confirmed=confirmed, params={**params, **board_params}, stackup=stack, driver={})
    fraction, z0, z0_tol = si_choices(ctx)
    decl = template.si_declarations(ir, ctx)
    si = si_constraints(ctx, decl, fraction, z0, z0_tol)
    plan.choices.extend(ctx.choices)
    board_params.update(ctx.new_params)
    plan.computed.extend(ctx.computed)
    plan.board.extend(class_line(c) for c in si.net_classes)
    plan.board.extend(path_line(p) for p in si.timing_paths)
    plan.board.extend(decl.lines)
    clash = sorted(k for k in board_params if k in params)
    if clash:
        return f"board parameters {clash} clash with the template's own"
    param_changes = [DesignChange(description=f"parameter {k}", target=f"parameters.{k}", operation="set", payload=v) for k, v in board_params.items()]
    plan.changes.extend([*param_changes, *changes, DesignChange(description="signal-integrity classes", target="si", operation="set", payload=si,
                                                                 rationale=f"template {t}: need-driven SI classes")])
    return None


__all__ = [
    "CONTROLLED_CLASS",
    "CRITICAL_FRACTION",
    "C_LOAD_F",
    "DEFAULT_CLASS",
    "PLANE_CLEARANCE_KEY",
    "RINGING_TOL_REL",
    "R_DRIVE_OHM",
    "T_RISE_S",
    "Z0_TARGET_OHM",
    "Z0_TOL_REL",
    "VALIDATION_ONLY",
    "BoardContext",
    "SIDeclarations",
    "add_board",
    "class_line",
    "design_file_layout_problem",
    "design_file_silk_text",
    "has_layout",
    "keepout_line",
    "layer_policy_refusal",
    "layout_fields",
    "layout_lines",
    "layout_onto",
    "path_line",
    "proposed_silk_text",
    "read_board_layers",
    "si_choices",
    "si_constraints",
    "template_silk_text",
]
