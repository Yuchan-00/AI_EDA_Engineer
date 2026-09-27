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

The stack: ``pcb_layers`` (alias ``layer_count``) read by
:func:`~ai_eda.design.inputs.read_layer_count`; without it the default 2
layers, shown as a choice. A 4-layer stack puts the ground plane on
``In1.Cu`` and the template's supply net (:attr:`Template.plane_nets`) on
``In2.Cu``, and adds the plane edge clearance as the parameter
:data:`PLANE_CLEARANCE_KEY` (the PCB agent draws the plane zones from it).
An IR that already carries a stack (a fab's, the user's) keeps it: the
template proposes no stack then and says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ai_eda.ir import CircuitIR, NetClass, PCBDesign, Provenance, SIConstraints, Stackup, TimingPath, Traced

from ai_eda.design.base import Choice, DesignChange, Plan, choice_provenance, structural_provenance
from ai_eda.design.inputs import LAYER_COUNT_KEY, DesignInput, read_layer_count
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
    if c.description and not c.default and c.target_z0_ohm is None:
        parts.append(c.description)  # a template class says in words why it is there (and, without an edge, why no rule applies)
    return "; ".join(parts)


def path_line(p: TimingPath) -> str:
    known = [k for k in ("t_co_max_s", "t_co_min_s", "t_su_min_s", "t_h_min_s") if getattr(p, k) is not None]
    return (
        f"timing path {p.name}: clock {p.clock_net} -> data {', '.join(p.data_nets)} ({p.direction}); f_clk {_fmt(p.f_clk_hz, 'Hz')}, "
        f"capture after {_fmt(p.capture_fraction)} of the period; datasheet terms " + (", ".join(f"{k} <- {v}" for k, v in sorted(p.terms_from.items())) or "-")
        + (f"; given: {', '.join(known)}" if known else "; none grounded yet (setup / hold margins stay NOT_VERIFIED until they are)")
    )


def add_board(template: Template, ir: CircuitIR, plan: Plan, *, confirmed: bool) -> str | None:
    """Append the board's stack and SI classes to a buildable ``plan`` (choices, inputs, table lines, changes); the refusal reason otherwise."""
    t = template.id
    layers, why = read_layer_count(ir)
    if layers is None:
        return f"{LAYER_COUNT_KEY}: {why}"
    params = _plan_params(plan)
    board_params: dict[str, Traced] = {}
    changes: list[DesignChange] = []
    existing = ir.pcb.stackup if ir.pcb is not None else None
    ground, power = template.plane_nets
    power = power or ground
    stack: Stackup | None
    if existing is not None:
        stack = existing
        plan.board.append(f"stackup: the IR's own {existing.layer_count}-layer stack is kept (no generic stack proposed)")
    else:
        plan.choices.extend(stackup_choices(layers, ground_net=ground, power_net=power))
        stack = generic_stackup(layers.value, t, confirmed=confirmed, ground_net=ground, power_net=power)
        if layers.traced is not None:  # the count is the user's: the stack's provenance says which requirement it serves
            inp = DesignInput(key=LAYER_COUNT_KEY, requirement=layers.requirement, traced=layers.traced)  # type: ignore[arg-type]
            plan.inputs[LAYER_COUNT_KEY] = inp
            board_params[LAYER_COUNT_KEY] = layers.traced
            prov = stack.provenance.model_copy(update={"derived_from": [layers.requirement.id]})  # type: ignore[union-attr]
            stack = stack.model_copy(update={"provenance": prov})
        if layers.value == 4:
            board_params[PLANE_CLEARANCE_KEY] = Traced(
                value=PLANE_EDGE_CLEARANCE_MM, unit="mm",
                provenance=choice_provenance(t, f"stackup.plane_edge_clearance = {PLANE_EDGE_CLEARANCE_MM!r} mm: plane zones end this far inside the board edge", confirmed),
            )
        if ir.pcb is None:
            changes.append(DesignChange(description=f"board stack: {layers.value} layers", target="pcb", operation="set",
                                        payload=PCBDesign(layers=board_layers(stack), stackup=stack), rationale=f"template {t}: generic {layers.value}-layer stack"))
        else:
            changes.append(DesignChange(description=f"board layers: {layers.value}", target="pcb.layers", operation="set", payload=board_layers(stack),
                                        rationale=f"template {t}: generic {layers.value}-layer stack"))
            changes.append(DesignChange(description=f"board stack: {layers.value} layers", target="pcb.stackup", operation="set", payload=stack,
                                        rationale=f"template {t}: generic {layers.value}-layer stack"))
        plan.board.append(f"stackup: generic {layers.value}-layer ({'from ' + layers.requirement.id if layers.requirement is not None else 'the default'}); "
                          + ("planes In1.Cu = " + ground + ", In2.Cu = " + power if layers.value == 4 else "no plane: impedance is undefined, only delay bounds are computed"))
    ctx = BoardContext(template_id=t, confirmed=confirmed, params={**params, **board_params}, stackup=stack, driver={})
    ctx.driver = {
        "t_rise_s": ctx.choice("si.t_rise", T_RISE_S, "s", (
            "driver edge for the critical-length rule and the SPICE check: 1 ns, conservative (faster than the parts are expected to switch; "
            "not a measured value - a grounded datasheet t_rise of the driver replaces it)")),
        "r_drive_ohm": ctx.choice("si.r_drive", R_DRIVE_OHM, "ohm", "driver source resistance of the SPICE check (a typical CMOS output at 5 V; not a datasheet value)"),
        "c_load_f": ctx.choice("si.c_load", C_LOAD_F, "F", "far-end load of the SPICE check (a typical CMOS input with its pad; not a datasheet value)"),
        "ringing_tol_rel": ctx.choice("si.ringing_tol", RINGING_TOL_REL, None, "overshoot / undershoot / settling band of the SPICE check (15 % of the swing)"),
    }
    fraction = ctx.choice("si.critical_fraction", CRITICAL_FRACTION, None, (
        "critical-length rule l_crit = fraction x t_r / t_pd: 1/2 (the round trip is shorter than the edge); a routed net longer than l_crit "
        f"is promoted to {CONTROLLED_CLASS}"))
    z0 = ctx.choice("si.z0_target", Z0_TARGET_OHM, "ohm", f"target impedance of the controlled class {CONTROLLED_CLASS} (defined only over a reference plane)")
    z0_tol = ctx.choice("si.z0_tol", Z0_TOL_REL, None, f"tolerance of the {CONTROLLED_CLASS} impedance (10 %)")
    decl = template.si_declarations(ir, ctx)
    drv = ctx.driver
    default = NetClass(
        name=DEFAULT_CLASS, default=True, description="every net no other class lists: routed at the board rules unless the critical-length rule promotes it",
        t_rise_s=drv["t_rise_s"], r_drive_ohm=drv["r_drive_ohm"], c_load_f=drv["c_load_f"], ringing_tol_rel=drv["ringing_tol_rel"], driver=decl.driver,
        promote_to=CONTROLLED_CLASS, provenance=ctx.structural("default net class"),
    )
    controlled = NetClass(
        name=CONTROLLED_CLASS, description="controlled impedance for nets the critical-length rule finds electrically long",
        target_z0_ohm=z0, z0_tol_rel=z0_tol, t_rise_s=drv["t_rise_s"], r_drive_ohm=drv["r_drive_ohm"], c_load_f=drv["c_load_f"],
        ringing_tol_rel=drv["ringing_tol_rel"], driver=decl.driver, provenance=ctx.structural("controlled-impedance class for promoted nets"),
    )
    si = SIConstraints(net_classes=[default, controlled, *decl.classes], timing_paths=list(decl.timing_paths), critical_fraction=fraction,
                       provenance=ctx.structural("signal-integrity classes: need-driven (declared interfaces + the critical-length rule)"))
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
    "BoardContext",
    "SIDeclarations",
    "add_board",
    "class_line",
    "path_line",
]
