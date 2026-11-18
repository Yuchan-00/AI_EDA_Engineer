"""The board stackups a template builds from: a generic 2-layer and 4-layer FR-4 stack as confirmed choices, and their plane zones.

Invariant: no number here is a fab fact. The generic stacks are a
template's free choices - "a generic value, not a fab's" - shown in the
``confirm_design`` table (:func:`stackup_choices`) and stamped like every
other choice (:func:`~ai_eda.design.base.choice_provenance`: an
``assumption`` until the table is confirmed, then ``user_requirement`` with
the choice note). A fab's real stack does not enter the IR in this
version: ``--fab-capability`` grounds a ``stackup`` block verbatim from the
vendor page (:mod:`ai_eda.tools.manufacturing.capability_file`) and reports
it, but no agent writes it into ``ir.pcb.stackup`` yet (the stack the router
used would have to be re-routed with it); the plane assignment is always the
design's decision, never the page's.

Two shapes (``pcb_layers``, :func:`~ai_eda.design.inputs.read_layer_count`;
a template builds the counts of its :class:`~ai_eda.design.base.LayerPolicy`
and defaults to the policy's count - 2 unless the template states another -
and says so in the table, :func:`layer_count_choice_text`):

* 2 layers - ``F.Cu`` / core / ``B.Cu``, no plane: every impedance number is
  NOT_VERIFIED "impedance is undefined without a reference plane".
* 4 layers - ``F.Cu`` / prepreg / ``In1.Cu`` (plane: the ground net) / core /
  ``In2.Cu`` (plane: a power net) / prepreg / ``B.Cu``. Signals are routed on
  ``F.Cu`` / ``B.Cu`` only (vias through all four layers); each outer layer
  is a microstrip over the adjacent plane. The planes are IR zones
  (:func:`plane_zones`: the outline inset by the plane edge clearance, one
  zone per plane layer, the plane's net) that KiCad fills; connectivity
  through a zone stays NOT_VERIFIED, so every net is still routed with tracks
  - the planes are return paths and impedance references.

Everything is a pure function of its arguments (no clock, no randomness): the
same count, template and confirmation give the same stack and the same zones.

These stacks are FR-4. A board whose device needs another base material (a
high-frequency laminate, polyimide flex, an aluminium IMS) gets that
material's own stack, every number read from its data sheet
(:mod:`ai_eda.design.materials`); a board that names no material and no
material need keeps the stacks here byte for byte.

The layer escalation (:data:`LAYER_ESCALATION`, ``design.stackup.escalation``
0.1) has its own stacks, outside the template tables: the 4-layer entry above
plus :data:`ESCALATION_STACKS` for 6, 8, 10 and 12 copper layers, each with
its plane roles, the preferred direction of every signal layer and the via
structures the router may use. Every number is a generic choice and every
rule they obey names its source (:func:`stack_problems`,
:func:`via_drill_problems`); :func:`escalation_stackup` stamps them as
assumptions, never as a confirmed table row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from ai_eda.ir import BoardOutline, Layer, Provenance, ProvenanceKind, Stackup, StackupCopper, StackupDielectric, Traced, Zone
from ai_eda.ir.pcb import DielectricKind

from ai_eda.design.base import Choice, LayerPolicy, choice_provenance
from ai_eda.design.inputs import DEFAULT_LAYER_COUNT, LAYER_COUNT_KEY, LAYER_COUNT_OPTIONS, LayerCountInput

#: provenance ``tool`` of the zones :func:`plane_zones` derives
STACKUP_TOOL = "design.stackup"
#: bumped when a generic stack or the zone rule changes
STACKUP_VERSION = "0.1"
#: the plane zones' inset from the board edge (mm) unless the caller passes another: a choice, shown in the table
PLANE_EDGE_CLEARANCE_MM = 0.5
#: what every generic number's choice text ends with
GENERIC_NOTE = "a generic value, not a fab's (a fab's stack is not read into the IR in this version; only the fab's measured impedance is real)"


#: what a plane layer of a generic stack carries: the board's ground net or its power net
PlaneRole = Literal["ground", "power"]
#: a signal layer's preferred routing direction
Direction = Literal["horizontal", "vertical"]


@dataclass(frozen=True)
class GenericStack:
    """A generic stack's numbers: copper ``(name, thickness um)`` top to bottom, dielectrics ``(kind, thickness mm, er, Hz)`` between them.

    Only the layer escalation reads the last four fields (the template path,
    :func:`generic_stackup`, places its planes itself): ``planes`` names each
    plane layer's role, ``vias`` the via structures besides the through via
    ``F.Cu``-``B.Cu`` (always there) as ``(kind, top, bottom)`` with kind
    ``blind`` / ``buried`` / ``micro``, ``directions`` each signal layer's
    preferred direction in stack order, ``roles_source`` the document the
    layer-role template comes from.
    """

    layer_count: int
    title: str
    copper: tuple[tuple[str, float], ...]
    dielectrics: tuple[tuple[DielectricKind, float, float, float], ...]
    planes: tuple[tuple[str, PlaneRole], ...] = ()
    vias: tuple[tuple[str, str, str], ...] = ()
    directions: tuple[tuple[str, Direction], ...] = ()
    roles_source: str = ""

    @property
    def thickness_mm(self) -> float:
        return sum(d[1] for d in self.dielectrics) + sum(c[1] for c in self.copper) / 1000.0

    def names(self) -> list[str]:
        return [name for name, _ in self.copper]

    def signal_layers(self) -> list[str]:
        """The copper layers that are not planes, in stack order."""
        planes = {name for name, _ in self.planes}
        return [name for name, _ in self.copper if name not in planes]


#: TI's suggested stackups (opened 2026-11-12; 6, 8 and 10 layers, no 12-layer entry)
TI_STACKUPS = "TI SPRAAR7J 'High-Speed Interface Layout Guidelines' (Nov 2018, rev. Feb 2023) sec. 3.12 Table 3-1"
#: In-Circuit Design's stackup note (opened 2026-11-12 from a copy on cdn.hackaday.io; the stackups are the figures' tables)
ICD_STACKUPS = "B. Olney, In-Circuit Design AN2011_2 'Multilayer PCB Stackup Planning' (2011)"

_ICD_4 = f"{ICD_STACKUPS} Fig. 2 (Top / GND / VCC / Bottom)"


#: the generic stacks, 1.6 mm FR-4 each (er 4.5 stated at 1 MHz)
GENERIC_STACKS: dict[int, GenericStack] = {
    2: GenericStack(
        layer_count=2, title="2-layer 1.6 mm FR-4",
        copper=(("F.Cu", 35.0), ("B.Cu", 35.0)),
        dielectrics=((DielectricKind.CORE, 1.53, 4.5, 1.0e6),),
    ),
    4: GenericStack(
        layer_count=4, title="4-layer 1.6 mm FR-4",
        copper=(("F.Cu", 35.0), ("In1.Cu", 17.5), ("In2.Cu", 17.5), ("B.Cu", 35.0)),
        dielectrics=((DielectricKind.PREPREG, 0.2, 4.5, 1.0e6), (DielectricKind.CORE, 1.095, 4.5, 1.0e6), (DielectricKind.PREPREG, 0.2, 4.5, 1.0e6)),
        # read by the layer escalation only (generic_stackup places the template's planes itself)
        planes=(("In1.Cu", "ground"), ("In2.Cu", "power")),
        vias=(("micro", "F.Cu", "In1.Cu"), ("micro", "In2.Cu", "B.Cu")),
        directions=(("F.Cu", "horizontal"), ("B.Cu", "vertical")),
        roles_source=_ICD_4,
    ),
}


def _stack(layer_count: int) -> GenericStack:
    if layer_count not in GENERIC_STACKS:
        raise ValueError(f"no generic stack for {layer_count} layers (one of {sorted(GENERIC_STACKS)})")
    return GENERIC_STACKS[layer_count]


def stack_description(layer_count: int) -> str:
    """One line naming every number of the generic stack (the choice text)."""
    g = _stack(layer_count)
    copper = ", ".join(f"{name} {um:g} um" for name, um in g.copper)
    diel = ", ".join(f"{kind.value} {mm:g} mm er {er:g} @ {hz / 1e6:g} MHz" for kind, mm, er, hz in g.dielectrics)
    return f"{g.title}: copper {copper}; dielectrics {diel} - {GENERIC_NOTE}"


def plane_description(ground_net: str, power_net: str) -> str:
    return (f"In1.Cu = {ground_net} plane, In2.Cu = {power_net} plane: return paths and the impedance reference of F.Cu / B.Cu "
            "(every net is still routed with tracks; the planes are zones KiCad fills)")


#: what each generic count gives the board, for the row that names the alternatives to a default
_LAYER_BOARD_TEXT: dict[int, str] = {2: "a 2-layer board without planes", 4: "a 4-layer board with ground / power planes"}


def layer_count_choice_text(policy: LayerPolicy | None = None) -> str:
    """The description of the ``pcb_layers`` row a template shows when no requirement states a count.

    Without a policy, or with the generic one (every generic stack, default
    :data:`~ai_eda.design.inputs.DEFAULT_LAYER_COUNT`), the text every
    template showed before policies existed, byte for byte (a confirmed
    table's hash stays valid). A template's own default names the counts the
    user may answer instead; a policy with one count says it is the only
    one; a restricting policy quotes its reason.
    """
    if policy is None or policy.is_generic:
        return f"board layer count: the default {DEFAULT_LAYER_COUNT} layers (answer {LAYER_COUNT_KEY}=4 for a 4-layer board with ground / power planes)"
    why = f": {policy.reason.strip()}" if policy.reason.strip() else ""
    others = [n for n in policy.allowed if n != policy.default]
    if not others:
        return f"board layer count: {policy.default} layers, the only count this template builds{why}"
    alternatives = "; ".join(f"answer {LAYER_COUNT_KEY}={n} for {_LAYER_BOARD_TEXT.get(n, f'a {n}-layer board')}" for n in others)
    return f"board layer count: this template's default {policy.default} layers ({alternatives}){why}"


def stackup_choices(layers: LayerCountInput, *, ground_net: str = "GND", power_net: str = "+5V",
                    edge_clearance_mm: float = PLANE_EDGE_CLEARANCE_MM, policy: LayerPolicy | None = None) -> list[Choice]:
    """The rows a template adds to its ``confirm_design`` table for the board stack.

    The layer count is a row only when no requirement stated it (the default
    is a choice the user must see; its text follows the template's ``policy``,
    :func:`layer_count_choice_text`); the stack's numbers always are, and a
    4-layer stack adds the plane assignment and the planes' edge clearance.
    """
    out: list[Choice] = []
    if layers.is_default:
        out.append(Choice(LAYER_COUNT_KEY, layer_count_choice_text(policy), layers.value, None))
    out.append(Choice("stackup", stack_description(layers.value)))
    if layers.value == 4:
        out.append(Choice("stackup.planes", plane_description(ground_net, power_net)))
        out.append(Choice("stackup.plane_edge_clearance", "plane zones end this far inside the board edge", edge_clearance_mm, "mm"))
    return out


def generic_stackup(layer_count: int, template_id: str, *, confirmed: bool, ground_net: str = "GND", power_net: str = "+5V") -> Stackup:
    """The generic stack for ``layer_count`` (2 or 4) with every number stamped as ``template_id``'s choice (confirmed or not)."""
    if layer_count not in LAYER_COUNT_OPTIONS:
        raise ValueError(f"no generic stack for {layer_count} layers (one of {list(LAYER_COUNT_OPTIONS)})")
    g = _stack(layer_count)
    description = stack_description(layer_count)

    def choice(value, unit: str | None) -> Traced:
        return Traced(value=value, unit=unit, provenance=choice_provenance(template_id, description, confirmed))

    planes = {"In1.Cu": ground_net, "In2.Cu": power_net} if layer_count == 4 else {}
    plane_text = plane_description(ground_net, power_net)
    copper = [
        StackupCopper(
            name=name, thickness_um=choice(um, "um"),
            plane_net=Traced(value=planes[name], provenance=choice_provenance(template_id, plane_text, confirmed)) if name in planes else None,
        )
        for name, um in g.copper
    ]
    dielectrics = [
        StackupDielectric(kind=kind, thickness_mm=choice(mm, "mm"), er=choice(er, None), er_frequency_hz=choice(hz, "Hz"))
        for kind, mm, er, hz in g.dielectrics
    ]
    return Stackup(copper=copper, dielectrics=dielectrics, provenance=choice_provenance(template_id, description, confirmed))


def with_planes(stackup: Stackup, planes: dict[str, Traced[str]], provenance: Provenance | None = None) -> Stackup:
    """``stackup`` with the plane nets ``planes`` (layer name -> traced net) assigned; a grounded fab stack gets its planes this way.

    ``ValueError`` for a layer the stack does not have. The other layers keep
    what they had; ``provenance`` (default: the stack's own) records who made
    the assignment.
    """
    names = set(stackup.copper_names())
    unknown = sorted(set(planes) - names)
    if unknown:
        raise ValueError(f"plane layer(s) {unknown} are not in the stackup {stackup.copper_names()}")
    copper = [c.model_copy(update={"plane_net": planes[c.name]}) if c.name in planes else c for c in stackup.copper]
    return Stackup(copper=copper, dielectrics=list(stackup.dielectrics), solder_mask=stackup.solder_mask, provenance=provenance or stackup.provenance,
                   material=stackup.material, metal_base=stackup.metal_base)


def board_layers(stackup: Stackup) -> list[Layer]:
    """``PCBDesign.layers`` for the stack: a plane layer is KiCad type ``power``, every other one ``signal``.

    A stack with one copper layer on a metal base (an IMS,
    :mod:`ai_eda.design.materials`) also gets ``B.Cu`` (signal): a KiCad board
    has an even copper layer count (the PCB compiler refuses an odd one), and
    the PCB agent keeps all copper off it with a keep-out.
    """
    out = [Layer(name=c.name, kind="power" if c.plane_net is not None else "signal") for c in stackup.copper]
    if stackup.metal_base is not None and "B.Cu" not in stackup.copper_names():
        out.append(Layer(name="B.Cu", kind="signal"))
    return out


def plane_zones(stackup: Stackup, outline: BoardOutline, edge_clearance: float | Traced[float] = PLANE_EDGE_CLEARANCE_MM) -> list[Zone]:
    """One zone per plane layer: the rectangular outline inset by ``edge_clearance`` mm, on that layer, of the plane's net.

    ``edge_clearance`` may be a ``Traced`` (its provenance kind and note are
    quoted in each zone's provenance note); ``ValueError`` when the inset
    leaves no area or the clearance is not a finite number >= 0. The zones'
    own provenance is ``derived`` by :data:`STACKUP_TOOL` from the plane
    assignment and the outline.
    """
    c = float(edge_clearance.value if isinstance(edge_clearance, Traced) else edge_clearance)
    if not math.isfinite(c) or c < 0.0:
        raise ValueError(f"plane edge clearance must be a finite number >= 0, got {c!r}")
    x0, y0 = float(outline.origin_x_mm) + c, float(outline.origin_y_mm) + c
    x1, y1 = float(outline.origin_x_mm) + float(outline.width_mm) - c, float(outline.origin_y_mm) + float(outline.height_mm) - c
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"a {outline.width_mm} x {outline.height_mm} mm outline inset by {c} mm leaves no plane area")
    source = f"; edge clearance {c:g} mm"
    if isinstance(edge_clearance, Traced):
        source += f" ({edge_clearance.provenance.kind.value}" + (f": {edge_clearance.provenance.note}" if edge_clearance.provenance.note else "") + ")"
    out: list[Zone] = []
    for layer in stackup.plane_layers():
        net = str(layer.plane_net.value)
        out.append(Zone(
            net=net, layer=layer.name, polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            provenance=Provenance(
                kind=ProvenanceKind.DERIVED, tool=STACKUP_TOOL, tool_version=STACKUP_VERSION,
                derived_from=[f"pcb.stackup.copper[{layer.name}].plane_net", "pcb.outline"],
                note=f"{net} plane on {layer.name}: the board outline inset by the plane edge clearance{source}",
            ),
        ))
    return out


# --------------------------------------------------------------------------- the layer escalation (design.stackup.escalation 0.1)

#: provenance ``tool`` / ``tool_version`` of a stack the layer escalation chose
ESCALATION_TOOL = "design.stackup.escalation"
ESCALATION_VERSION = "0.1"

#: the project owner's layer-escalation ladder (2026-11-12): start at 2 copper layers, then 4, 6, 8, 10, 12, and stop at 12.
#: The owner once typed "4,5,8,10,12"; 5 was taken as a typo for 6 (an odd copper count cannot be built: escalation_refusal)
LAYER_ESCALATION: tuple[int, ...] = (2, 4, 6, 8, 10, 12)


def _check_ladder(ladder: object) -> None:
    """``ValueError`` naming the problem unless ``ladder`` is a non-empty tuple of plain ints that starts at 2 and strictly ascends.

    Run on :data:`LAYER_ESCALATION` at import. An odd count is not refused
    here: :func:`escalation_refusal` refuses it at run time with its reason.
    """
    if not isinstance(ladder, tuple) or not ladder:
        raise ValueError(f"LAYER_ESCALATION must be a non-empty tuple of copper layer counts, got {ladder!r}")
    bad = [n for n in ladder if type(n) is not int]
    if bad:
        raise ValueError(f"LAYER_ESCALATION must hold plain ints, got {bad!r}")
    if ladder[0] != 2:
        raise ValueError(f"LAYER_ESCALATION must start at 2 (the board as it is routed before any escalation), got {ladder[0]}")
    if any(b <= a for a, b in zip(ladder, ladder[1:])):
        raise ValueError(f"LAYER_ESCALATION must be strictly ascending, got {ladder}")


_check_ladder(LAYER_ESCALATION)

#: Eurocircuits' rule for mechanical blind / buried via runs (sequential lamination), opened 2026-11-12
EUROCIRCUITS_VIA_RUNS = ("Eurocircuits Buildup Editor User Guide, 'Blind/Buried via runs' (served at eurocircuits.com/blog/blind-and-buried-vias/, "
                         "which research_synthesis cites as 'Blind and Buried Vias', D. Stans, 28 Sep 2016)")
#: the microvia definition: a blind structure of aspect ratio at most 1:1, at most 0.25 mm deep from the capture land foil to the target land
MICROVIA_SOURCE = "IPC-T-50M's microvia definition as quoted by NCAB 'Microvias' (opened 2026-11-12; the IPC text itself was not opened)"
MICROVIA_MAX_DEPTH_MM = 0.25
MICROVIA_MAX_ASPECT = 1.0
#: the through-via aspect ratio (hole depth = board thickness : drill): the upper end of the recommended 6:1 to 8:1
THROUGH_VIA_SOURCE = ("Wurth Elektronik 'HDI Design Guide' v1.2 (Sep 2018) p. 2, citing IPC-2221A/IPC-2222 (opened 2026-11-12; "
                      "the IPC texts were not opened)")
THROUGH_VIA_MAX_ASPECT = 8.0
#: the rule text every escalated stack's description carries for its via structures
VIA_SPAN_RULES = ("mechanical blind / buried runs follow Eurocircuits' rule: an even number of copper layers, never ending on a core's top "
                  "side nor starting on its bottom side, two runs nested or disjoint; micro vias cross exactly one dielectric, at most "
                  f"{MICROVIA_MAX_DEPTH_MM:g} mm deep per IPC-T-50M as NCAB quotes it")

_PP, _CORE = DielectricKind.PREPREG, DielectricKind.CORE


def _copper(n: int) -> tuple[tuple[str, float], ...]:
    """``n`` copper layers: 35 um outside, 17.5 um inside (the 4-layer entry's copper, carried over)."""
    return (("F.Cu", 35.0), *((f"In{i}.Cu", 17.5) for i in range(1, n - 1)), ("B.Cu", 35.0))


def _fr4(kind: DielectricKind, mm: float) -> tuple[DielectricKind, float, float, float]:
    """One FR-4 dielectric, er 4.5 stated at 1 MHz (the 2- and 4-layer entries' material, carried over)."""
    return (kind, mm, 4.5, 1.0e6)


#: The escalation stacks above 4 layers - every number a generic choice, not a fab's. 1.6 mm FR-4 (er 4.5 at 1 MHz), copper
#: 35 um outside and 17.5 um inside, as the 4-layer entry; a 0.1 mm build-up prepreg outermost on each side, and inside it
#: prepreg and core alternate, starting and ending with prepreg (mirror-symmetric). Each total is 1.6 mm. The layer roles
#: follow ``roles_source``; ``vias`` is what :func:`via_structures` generates (pinned by tests/test_stack_family.py), and
#: ``directions`` alternate horizontal / vertical down the signal layers, so two signal layers with no plane between them
#: are routed at 90 degrees (TI SPRAAR7J Table 3-1 note 1; ICD AN2011_2 p. 5).
ESCALATION_STACKS: dict[int, GenericStack] = {
    6: GenericStack(
        layer_count=6, title="6-layer 1.6 mm FR-4", copper=_copper(6),
        dielectrics=(_fr4(_PP, 0.1), _fr4(_PP, 0.2), _fr4(_CORE, 0.86), _fr4(_PP, 0.2), _fr4(_PP, 0.1)),
        planes=(("In1.Cu", "ground"), ("In4.Cu", "power")),
        vias=(("micro", "F.Cu", "In1.Cu"), ("micro", "In4.Cu", "B.Cu"), ("blind", "F.Cu", "In3.Cu"), ("buried", "In2.Cu", "In3.Cu")),
        directions=(("F.Cu", "horizontal"), ("In2.Cu", "vertical"), ("In3.Cu", "horizontal"), ("B.Cu", "vertical")),
        roles_source=(f"{TI_STACKUPS} 6-layer: signal / ground / signal / signal / power-ground / signal, the power-ground plane taken as "
                      f"the power plane as {ICD_STACKUPS} Fig. 3 has it (VCC)"),
    ),
    8: GenericStack(
        layer_count=8, title="8-layer 1.6 mm FR-4", copper=_copper(8),
        dielectrics=(_fr4(_PP, 0.1), _fr4(_PP, 0.2), _fr4(_CORE, 0.3125), _fr4(_PP, 0.2), _fr4(_CORE, 0.3125), _fr4(_PP, 0.2), _fr4(_PP, 0.1)),
        planes=(("In1.Cu", "ground"), ("In4.Cu", "power"), ("In6.Cu", "ground")),
        vias=(("micro", "F.Cu", "In1.Cu"), ("micro", "In6.Cu", "B.Cu"), ("blind", "F.Cu", "In3.Cu"), ("blind", "In4.Cu", "B.Cu"),
              ("buried", "In2.Cu", "In3.Cu")),
        directions=(("F.Cu", "horizontal"), ("In2.Cu", "vertical"), ("In3.Cu", "horizontal"), ("In5.Cu", "vertical"), ("B.Cu", "horizontal")),
        roles_source=(f"{TI_STACKUPS} 8-layer: signal / ground / signal / signal / power-ground / signal / ground / signal, the power-ground "
                      f"plane taken as the power plane (5 signal layers; the alternative, {ICD_STACKUPS} Fig. 4 'risk aversive' "
                      "signal / ground / signal / power / ground / signal / power / signal, has 4)"),
    ),
    10: GenericStack(
        layer_count=10, title="10-layer 1.6 mm FR-4", copper=_copper(10),
        dielectrics=(_fr4(_PP, 0.1), _fr4(_PP, 0.14), _fr4(_CORE, 0.21), _fr4(_PP, 0.14), _fr4(_CORE, 0.21), _fr4(_PP, 0.14), _fr4(_CORE, 0.21),
                     _fr4(_PP, 0.14), _fr4(_PP, 0.1)),
        planes=(("In1.Cu", "ground"), ("In4.Cu", "power"), ("In5.Cu", "ground"), ("In8.Cu", "ground")),
        vias=(("micro", "F.Cu", "In1.Cu"), ("micro", "In8.Cu", "B.Cu"), ("blind", "F.Cu", "In3.Cu"), ("blind", "In6.Cu", "B.Cu"),
              ("buried", "In2.Cu", "In3.Cu"), ("buried", "In6.Cu", "In7.Cu")),
        directions=(("F.Cu", "horizontal"), ("In2.Cu", "vertical"), ("In3.Cu", "horizontal"), ("In6.Cu", "vertical"), ("In7.Cu", "horizontal"),
                    ("B.Cu", "vertical")),
        roles_source=(f"{TI_STACKUPS} 10-layer: signal / ground / signal / signal / power / power-ground / signal / signal / ground / signal, "
                      f"the power-ground plane taken as ground as {ICD_STACKUPS} Fig. 5 has it (whose ninth layer is VCC where TI has ground)"),
    ),
    12: GenericStack(
        layer_count=12, title="12-layer 1.6 mm FR-4", copper=_copper(12),
        dielectrics=(_fr4(_PP, 0.1), _fr4(_PP, 0.115), _fr4(_CORE, 0.145), _fr4(_PP, 0.115), _fr4(_CORE, 0.145), _fr4(_PP, 0.115),
                     _fr4(_CORE, 0.145), _fr4(_PP, 0.115), _fr4(_CORE, 0.145), _fr4(_PP, 0.115), _fr4(_PP, 0.1)),
        planes=(("In1.Cu", "ground"), ("In4.Cu", "power"), ("In7.Cu", "ground"), ("In10.Cu", "power")),
        vias=(("micro", "F.Cu", "In1.Cu"), ("micro", "In10.Cu", "B.Cu"), ("blind", "F.Cu", "In3.Cu"), ("blind", "In8.Cu", "B.Cu"),
              ("buried", "In2.Cu", "In3.Cu"), ("buried", "In4.Cu", "In7.Cu"), ("buried", "In8.Cu", "In9.Cu")),
        directions=(("F.Cu", "horizontal"), ("In2.Cu", "vertical"), ("In3.Cu", "horizontal"), ("In5.Cu", "vertical"), ("In6.Cu", "horizontal"),
                    ("In8.Cu", "vertical"), ("In9.Cu", "horizontal"), ("B.Cu", "vertical")),
        roles_source=(f"{ICD_STACKUPS} Fig. 6: Top / GND / signal / signal / VDD / signal / signal / GND / signal / signal / VCC / Bottom, "
                      "both supply planes taken as the power plane (TI Table 3-1 has no 12-layer stack)"),
    ),
}


def _index(g: GenericStack, name: str) -> int:
    names = g.names()
    if name not in names:
        raise ValueError(f"{name!r} is not a copper layer of the {g.title} stack {names}")
    return names.index(name)


def _core_faces(g: GenericStack) -> tuple[set[int], set[int]]:
    """Copper indices on a core's top side and on a core's bottom side."""
    cores = [i for i, d in enumerate(g.dielectrics) if d[0] is DielectricKind.CORE]
    return set(cores), {i + 1 for i in cores}


def mechanical_span_problem(g: GenericStack, top: str, bottom: str) -> str | None:
    """Why a drilled (mechanical) via run from ``top`` down to ``bottom`` cannot be built in ``g``, or ``None``.

    Eurocircuits' rule (:data:`EUROCIRCUITS_VIA_RUNS`), applied as stated to
    the stack's core / prepreg map: a run goes through an even number of
    copper layers, never ends on the top side of a core and never starts on
    the bottom side of one. A copper layer between two prepregs (the build-up
    layers ``In1.Cu`` / ``In<N-2>.Cu`` of the escalation stacks) is neither, so
    the rule as stated does not restrict it. Whether a fab can press the
    resulting lamination sequence is not checked (``mfg.capability`` reports
    blind / buried / micro via processes NOT_VERIFIED).
    """
    a, b = _index(g, top), _index(g, bottom)
    if not a < b:
        return f"{top} is not above {bottom}"
    tops, bottoms = _core_faces(g)
    count = b - a + 1
    if count % 2:
        return f"goes through {count} copper layers - a via run must go through an even number ({EUROCIRCUITS_VIA_RUNS})"
    if a in bottoms:
        return f"starts on {top}, the bottom side of a core - a via run cannot ({EUROCIRCUITS_VIA_RUNS})"
    if b in tops:
        return f"ends on {bottom}, the top side of a core - a via run cannot ({EUROCIRCUITS_VIA_RUNS})"
    return None


def legal_spans(g: GenericStack) -> list[tuple[str, str]]:
    """Every ``(top, bottom)`` a mechanical via run may span in ``g`` (:func:`mechanical_span_problem`), top end first, then the shorter run."""
    names = g.names()
    return [(names[a], names[b]) for a in range(len(names)) for b in range(a + 1, len(names))
            if mechanical_span_problem(g, names[a], names[b]) is None]


def microvia_depth_mm(g: GenericStack, top: str, bottom: str) -> float:
    """The depth of a micro via between two adjacent copper layers: the dielectric plus the thicker end's copper.

    IPC-T-50M (as NCAB quotes it) measures from the capture land foil to the
    target land; counting the thicker of the two copper layers as the capture
    foil is a conservative stated reading.
    """
    a, b = _index(g, top), _index(g, bottom)
    if b != a + 1:
        raise ValueError(f"{top}-{bottom} is not one dielectric of the {g.title} stack")
    return g.dielectrics[a][1] + max(g.copper[a][1], g.copper[b][1]) / 1000.0


def span_problem(g: GenericStack, kind: str, top: str, bottom: str) -> str | None:
    """Why a via of ``kind`` from ``top`` down to ``bottom`` is not a legal structure of ``g``, or ``None``.

    * ``through`` joins ``F.Cu`` and ``B.Cu``.
    * ``micro`` crosses exactly one dielectric (a laser microvia, a stated
      rule of this version) at most :data:`MICROVIA_MAX_DEPTH_MM` deep
      (:func:`microvia_depth_mm`; :data:`MICROVIA_SOURCE`). It is laser
      drilled, so Eurocircuits' rule for drilled runs does not apply to it.
    * ``blind`` has exactly one outer end, ``buried`` none; both are drilled
      runs (:func:`mechanical_span_problem`).
    """
    names = g.names()
    for name in (top, bottom):
        if name not in names:
            return f"{name} is not a copper layer of the {g.title} stack"
    a, b = names.index(top), names.index(bottom)
    if not a < b:
        return f"{top} is not above {bottom}"
    outer = (a == 0) + (b == len(names) - 1)
    if kind == "through":
        return None if outer == 2 else "a through via joins F.Cu and B.Cu"
    if kind == "micro":
        if b != a + 1:
            return "a micro via crosses exactly one dielectric (joins two adjacent copper layers)"
        depth = microvia_depth_mm(g, top, bottom)
        if depth > MICROVIA_MAX_DEPTH_MM:
            return f"a micro via is at most {MICROVIA_MAX_DEPTH_MM:g} mm deep, this one {depth:g} mm ({MICROVIA_SOURCE})"
        return None
    if kind == "blind":
        if outer != 1:
            return "a blind via has exactly one outer end"
        return mechanical_span_problem(g, top, bottom)
    if kind == "buried":
        if outer != 0:
            return "a buried via has no outer end"
        return mechanical_span_problem(g, top, bottom)
    return f"via kind {kind!r} is not one of through, blind, buried, micro"


def _nested_or_disjoint(s: tuple[int, int], t: tuple[int, int]) -> bool:
    """Two runs (copper index ranges) share no copper layer, or one lies completely inside the other (Eurocircuits)."""
    return s[1] < t[0] or t[1] < s[0] or (s[0] <= t[0] and t[1] <= s[1]) or (t[0] <= s[0] and s[1] <= t[1])


def via_structures(g: GenericStack) -> tuple[tuple[str, str, str], ...]:
    """The via structures (besides the through via) the one escalation rule gives ``g``: ``(kind, top, bottom)`` in admission order.

    Candidates, in this order, each kept only when it is legal
    (:func:`span_problem`), not already kept, and nested in or disjoint from
    every run kept before it (the through via first) - Eurocircuits: a run
    cannot start or end inside, or at the end of, another unless one is
    completely enclosed in the other (each nesting costs a press cycle):

    1. ``micro`` over the outermost dielectric on each side (``F.Cu``-``In1.Cu``,
       then ``In<N-2>.Cu``-``B.Cu``): plane vias for outer-layer pads;
    2. ``blind`` from each outer layer (``F.Cu`` side first): the shortest
       legal run that holds at least two signal layers;
    3. ``buried`` for each pair of signal layers with no plane between them,
       top to bottom: the shortest legal run with no outer end that holds both
       (ties: the upper one).

    A stack without planes gets none. On the 6-layer stack the ``B.Cu`` side's
    blind run ``In2.Cu``-``B.Cu`` overlaps the ``F.Cu`` side's ``F.Cu``-``In3.Cu``
    without nesting, so it is dropped: an odd copper count from an outer layer
    is never legal, so no blind run reaches ``In2.Cu`` from ``F.Cu`` alone.
    """
    names = g.names()
    n = len(names)
    if not g.planes or n < 4:
        return ()
    signal = {names.index(s) for s in g.signal_layers()}
    spans = [(names.index(t), names.index(b)) for t, b in legal_spans(g)]

    def holds(a: int, b: int) -> int:
        return sum(1 for i in range(a, b + 1) if i in signal)

    candidates: list[tuple[str, int, int]] = [("micro", 0, 1), ("micro", n - 2, n - 1)]
    for side in ("top", "bottom"):
        runs = sorted((b - a, a, b) for a, b in spans
                      if (a == 0) != (b == n - 1) and (a == 0 if side == "top" else b == n - 1) and holds(a, b) >= 2)
        if runs:
            candidates.append(("blind", runs[0][1], runs[0][2]))
    for i in range(1, n - 2):
        if i in signal and i + 1 in signal:
            runs = sorted((b - a, a, b) for a, b in spans if 0 < a <= i and i + 1 <= b < n - 1)
            if runs:
                candidates.append(("buried", runs[0][1], runs[0][2]))
    kept: list[tuple[int, int]] = [(0, n - 1)]
    out: list[tuple[str, str, str]] = []
    for kind, a, b in candidates:
        if (a, b) in kept or span_problem(g, kind, names[a], names[b]) is not None:
            continue
        if all(_nested_or_disjoint((a, b), k) for k in kept):
            kept.append((a, b))
            out.append((kind, names[a], names[b]))
    return tuple(out)


def stack_problems(g: GenericStack) -> list[str]:
    """Every sourced stack rule ``g`` breaks (empty: none). For a stack with planes - the 2-layer board has none and is not checked.

    * Every signal layer is adjacent to a reference plane, at most two signal
      layers lie between two planes and each outer side carries one signal
      layer (ICD AN2011_2 p. 2; TI SPRAAR7J Table 3-1).
    * Each signal layer has one preferred direction, and two signal layers with
      no plane between them have different ones - routed at a 90 degree offset
      (TI SPRAAR7J Table 3-1 note 1; ICD AN2011_2 p. 5).
    * The build is mirror-symmetric: copper thickness and dielectric kind,
      thickness and er of mirrored layers are equal - a stated choice of these
      generic stacks (research_synthesis cites JLCPCB's mirrored impedance
      stacks, not opened in this phase). The copper balance of mirrored layers
      is a per-design quantity and is not checked here.
    * Every via structure is legal (:func:`span_problem`), the through via is
      implicit, and any two runs (through and micro included) are nested or
      disjoint (Eurocircuits).
    """
    names = g.names()
    n = len(names)
    out: list[str] = []
    if g.layer_count != n or len(g.dielectrics) != n - 1:
        return [f"{g.title}: {n} copper layers with {len(g.dielectrics)} dielectrics for layer_count {g.layer_count}"]
    roles: dict[str, str] = {}
    for layer, role in g.planes:
        if layer not in names:
            out.append(f"plane {layer} is not a copper layer of the stack")
        elif layer in roles:
            out.append(f"plane {layer} is listed twice")
        elif role not in ("ground", "power"):
            out.append(f"plane {layer}: role {role!r} is not ground or power")
        roles[layer] = role
    if not roles:
        return out + ["no plane layer: the stack rules are for a stack with reference planes"]
    plane = [name in roles for name in names]
    for i, name in enumerate(names):
        if not plane[i] and not ((i > 0 and plane[i - 1]) or (i < n - 1 and plane[i + 1])):
            out.append(f"signal layer {name} is not adjacent to a reference plane ({ICD_STACKUPS} p. 2)")
    runs: list[list[int]] = [[]]
    for i in range(n):
        if plane[i]:
            runs.append([])
        else:
            runs[-1].append(i)
    if len(runs[0]) != 1 or len(runs[-1]) != 1:
        out.append(f"each outer side carries exactly one signal layer before its first plane ({ICD_STACKUPS} p. 2)")
    for run in runs[1:-1]:
        if len(run) > 2:
            out.append(f"{len(run)} signal layers {[names[i] for i in run]} between two planes - at most two ({ICD_STACKUPS} p. 2)")
    signal = g.signal_layers()
    direction = dict(g.directions)
    if [layer for layer, _ in g.directions] != signal:
        out.append(f"preferred directions must name the signal layers {signal} once each in stack order, got {[layer for layer, _ in g.directions]}")
    out += [f"{layer}: preferred direction {d!r} is not horizontal or vertical" for layer, d in g.directions if d not in ("horizontal", "vertical")]
    for i in range(n - 1):
        upper, lower = names[i], names[i + 1]
        if not plane[i] and not plane[i + 1] and direction.get(upper) is not None and direction.get(upper) == direction.get(lower):
            out.append(f"adjacent signal layers {upper} and {lower} share the preferred direction {direction[upper]} - route them at a "
                       f"90 degree offset ({TI_STACKUPS} note 1)")
    for i in range(n // 2):
        if g.copper[i][1] != g.copper[n - 1 - i][1]:
            out.append(f"not mirror-symmetric: {names[i]} {g.copper[i][1]:g} um against {names[n - 1 - i]} {g.copper[n - 1 - i][1]:g} um")
    for i in range((n - 1) // 2):
        if g.dielectrics[i] != g.dielectrics[n - 2 - i]:
            out.append(f"not mirror-symmetric: dielectric {i} {g.dielectrics[i]} against dielectric {n - 2 - i} {g.dielectrics[n - 2 - i]}")
    kept: list[tuple[str, tuple[int, int]]] = [("through F.Cu-B.Cu", (0, n - 1))]
    for kind, top, bottom in g.vias:
        label = f"{kind} {top}-{bottom}"
        if kind == "through":
            out.append(f"{label}: the through via F.Cu-B.Cu is implicit, not listed")
            continue
        problem = span_problem(g, kind, top, bottom)
        if problem is not None:
            out.append(f"{label}: {problem}")
            continue
        span = (names.index(top), names.index(bottom))
        for other, s in kept:
            if s == span:
                out.append(f"{label}: the span of {other} again")
            elif not _nested_or_disjoint(span, s):
                out.append(f"{label} and {other} overlap without nesting - two via runs must be nested or disjoint ({EUROCIRCUITS_VIA_RUNS})")
        kept.append((label, span))
    return out


def via_drill_problems(g: GenericStack, drill_mm: float) -> list[str]:
    """The via aspect-ratio limits a drill of ``drill_mm`` breaks in ``g`` (empty: none).

    * The through via: board thickness : drill at most
      :data:`THROUGH_VIA_MAX_ASPECT` (:data:`THROUGH_VIA_SOURCE`). A blind or
      buried run is never deeper than the board, so it passes when the
      through via does.
    * Each micro via: depth (:func:`microvia_depth_mm`) : drill at most
      :data:`MICROVIA_MAX_ASPECT` (:data:`MICROVIA_SOURCE`).

    ``ValueError`` for a drill that is not a finite number > 0.
    """
    if isinstance(drill_mm, bool) or not isinstance(drill_mm, (int, float)) or not math.isfinite(drill_mm) or drill_mm <= 0.0:
        raise ValueError(f"a via drill must be a finite number > 0 mm, got {drill_mm!r}")
    out: list[str] = []
    thickness = g.thickness_mm
    if thickness / drill_mm > THROUGH_VIA_MAX_ASPECT:
        out.append(f"through via F.Cu-B.Cu: aspect ratio {thickness / drill_mm:.2f}:1 ({thickness:g} mm board, {drill_mm:g} mm drill) is above "
                   f"{THROUGH_VIA_MAX_ASPECT:g}:1 ({THROUGH_VIA_SOURCE})")
    for kind, top, bottom in g.vias:
        if kind == "micro":
            depth = microvia_depth_mm(g, top, bottom)
            if depth / drill_mm > MICROVIA_MAX_ASPECT:
                out.append(f"micro via {top}-{bottom}: aspect ratio {depth / drill_mm:.2f}:1 ({depth:g} mm deep, {drill_mm:g} mm drill) is above "
                           f"{MICROVIA_MAX_ASPECT:g}:1 ({MICROVIA_SOURCE})")
    return out


def escalation_refusal(n: object) -> str | None:
    """Why the ladder cannot use ``n`` copper layers, or ``None`` (2 - the board as it is - and 4, 6, 8, 10, 12).

    An odd count cannot be built: kicad-cli 10.0.6 refused to load a 5-copper-layer
    board (measured: "5 is not a valid layer number", Korean locale) and the PCB
    compiler refuses odd counts. A count without a generic stack - above 12, the
    owner's last rung - is refused with that reason.
    """
    if type(n) is not int:
        return f"{n!r} copper layers: refused - not a whole copper layer count"
    if n < 2:
        return f"{n} copper layers: refused - a board needs at least 2 copper layers"
    if n % 2:
        return (f"{n} copper layers: refused - KiCad builds only even copper layer counts (kicad-cli 10.0.6 refused a 5-layer board: "
                "'5 is not a valid layer number'; compilers/pcb.py refuses odd counts)")
    if n != 2 and n != 4 and n not in ESCALATION_STACKS:
        return f"{n} copper layers: refused - no generic stack for {n} layers" + (" (the project owner's ladder stops at 12)" if n > 12 else "")
    return None


def escalation_stack(n: int) -> GenericStack:
    """The escalation stack for ``n`` copper layers: :data:`GENERIC_STACKS` [4] for 4, else :data:`ESCALATION_STACKS` [n].

    ``ValueError`` with :func:`escalation_refusal`'s reason otherwise, and for
    2: the ladder's first rung is the board as it is, routed without a stack.
    """
    why = escalation_refusal(n)
    if why is not None:
        raise ValueError(why)
    if n == 2:
        raise ValueError("2 copper layers: the ladder's first rung is the board as it is - no escalation stack")
    return GENERIC_STACKS[4] if n == 4 else ESCALATION_STACKS[n]


def escalation_description(n: int, ground_net: str, power_net: str | None = None) -> str:
    """One line naming every number of the ``n``-layer escalation stack, its plane assignment, signal layers and via structures.

    The :func:`stack_description` format, then the planes (the ground net on
    the ground planes, ``power_net`` on the power planes - the ground net when
    the board has no power net), each signal layer's preferred direction, the
    via structures with their rule, and the layer-role source; it ends with
    :data:`GENERIC_NOTE`.
    """
    g = escalation_stack(n)
    copper = ", ".join(f"{name} {um:g} um" for name, um in g.copper)
    diel = ", ".join(f"{kind.value} {mm:g} mm er {er:g} @ {hz / 1e6:g} MHz" for kind, mm, er, hz in g.dielectrics)
    planes = ", ".join(
        f"{layer} = {ground_net} (ground)" if role == "ground"
        else f"{layer} = {power_net} (power)" if power_net is not None
        else f"{layer} = {ground_net} (a power plane position; the board has no power net)"
        for layer, role in g.planes
    )
    signal = ", ".join(f"{layer} {d}" for layer, d in g.directions)
    vias = ", ".join(["through F.Cu-B.Cu", *(f"{kind} {top}-{bottom}" for kind, top, bottom in g.vias)])
    return (f"{g.title}: copper {copper}; dielectrics {diel}; planes {planes}; signal layers (preferred direction) {signal}; "
            f"via structures {vias} ({VIA_SPAN_RULES}); layer roles after {g.roles_source} - {GENERIC_NOTE}")


def escalation_stackup(n: int, *, ground_net: str, power_net: str | None = None, note: str | None = None) -> Stackup:
    """The ``n``-layer escalation stack with its planes assigned, every number an assumption chosen by the owner's ladder.

    Every traced number and plane net is ``assumption`` with the note
    ``f"{escalation_description(...)} - chosen by the project owner's
    layer-escalation rule, not shown in a confirm_design table"``. The ground
    planes carry ``ground_net``, the power planes ``power_net`` (the ground
    net when it is ``None``). The stack-level provenance is a placeholder the
    PCB agent replaces with the escalation record; ``note`` overrides its
    text. ``ValueError`` for a count :func:`escalation_stack` refuses or a
    stack that breaks :func:`stack_problems`.
    """
    g = escalation_stack(n)
    problems = stack_problems(g)
    if problems:
        raise ValueError(f"the {g.title} stack breaks its rules: " + "; ".join(problems))
    text = f"{escalation_description(n, ground_net, power_net)} - chosen by the project owner's layer-escalation rule, not shown in a confirm_design table"

    def chosen(value, unit: str | None) -> Traced:
        return Traced(value=value, unit=unit, provenance=Provenance(kind=ProvenanceKind.ASSUMPTION, note=text))

    nets = {layer: ground_net if role == "ground" or power_net is None else power_net for layer, role in g.planes}
    copper = [StackupCopper(name=name, thickness_um=chosen(um, "um"), plane_net=chosen(nets[name], None) if name in nets else None)
              for name, um in g.copper]
    dielectrics = [StackupDielectric(kind=kind, thickness_mm=chosen(mm, "mm"), er=chosen(er, None), er_frequency_hz=chosen(hz, "Hz"))
                   for kind, mm, er, hz in g.dielectrics]
    placeholder = note if note is not None else (f"layer count {n}: a placeholder - the PCB agent replaces it with the layer-escalation record "
                                                 f"({ESCALATION_TOOL} {ESCALATION_VERSION})")
    return Stackup(copper=copper, dielectrics=dielectrics, provenance=Provenance(
        kind=ProvenanceKind.ASSUMPTION, tool=ESCALATION_TOOL, tool_version=ESCALATION_VERSION, note=placeholder))


__all__ = [
    "ESCALATION_STACKS",
    "ESCALATION_TOOL",
    "ESCALATION_VERSION",
    "EUROCIRCUITS_VIA_RUNS",
    "GENERIC_NOTE",
    "GENERIC_STACKS",
    "ICD_STACKUPS",
    "LAYER_ESCALATION",
    "MICROVIA_MAX_ASPECT",
    "MICROVIA_MAX_DEPTH_MM",
    "MICROVIA_SOURCE",
    "PLANE_EDGE_CLEARANCE_MM",
    "STACKUP_TOOL",
    "STACKUP_VERSION",
    "THROUGH_VIA_MAX_ASPECT",
    "THROUGH_VIA_SOURCE",
    "TI_STACKUPS",
    "VIA_SPAN_RULES",
    "Direction",
    "GenericStack",
    "PlaneRole",
    "board_layers",
    "escalation_description",
    "escalation_refusal",
    "escalation_stack",
    "escalation_stackup",
    "generic_stackup",
    "layer_count_choice_text",
    "legal_spans",
    "mechanical_span_problem",
    "microvia_depth_mm",
    "plane_description",
    "plane_zones",
    "span_problem",
    "stack_description",
    "stack_problems",
    "stackup_choices",
    "via_drill_problems",
    "via_structures",
    "with_planes",
]
