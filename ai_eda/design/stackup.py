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
the templates default to 2 and say so):

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
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ai_eda.ir import BoardOutline, Layer, Provenance, ProvenanceKind, Stackup, StackupCopper, StackupDielectric, Traced, Zone
from ai_eda.ir.pcb import DielectricKind

from ai_eda.design.base import Choice, choice_provenance
from ai_eda.design.inputs import DEFAULT_LAYER_COUNT, LAYER_COUNT_KEY, LAYER_COUNT_OPTIONS, LayerCountInput

#: provenance ``tool`` of the zones :func:`plane_zones` derives
STACKUP_TOOL = "design.stackup"
#: bumped when a generic stack or the zone rule changes
STACKUP_VERSION = "0.1"
#: the plane zones' inset from the board edge (mm) unless the caller passes another: a choice, shown in the table
PLANE_EDGE_CLEARANCE_MM = 0.5
#: what every generic number's choice text ends with
GENERIC_NOTE = "a generic value, not a fab's (a fab's stack is not read into the IR in this version; only the fab's measured impedance is real)"


@dataclass(frozen=True)
class GenericStack:
    """A generic stack's numbers: copper ``(name, thickness um)`` top to bottom, dielectrics ``(kind, thickness mm, er, Hz)`` between them."""

    layer_count: int
    title: str
    copper: tuple[tuple[str, float], ...]
    dielectrics: tuple[tuple[DielectricKind, float, float, float], ...]

    @property
    def thickness_mm(self) -> float:
        return sum(d[1] for d in self.dielectrics) + sum(c[1] for c in self.copper) / 1000.0


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


def stackup_choices(layers: LayerCountInput, *, ground_net: str = "GND", power_net: str = "+5V",
                    edge_clearance_mm: float = PLANE_EDGE_CLEARANCE_MM) -> list[Choice]:
    """The rows a template adds to its ``confirm_design`` table for the board stack.

    The layer count is a row only when no requirement stated it (the default
    is a choice the user must see); the stack's numbers always are, and a
    4-layer stack adds the plane assignment and the planes' edge clearance.
    """
    out: list[Choice] = []
    if layers.is_default:
        out.append(Choice(LAYER_COUNT_KEY, f"board layer count: the default {DEFAULT_LAYER_COUNT} layers "
                                           f"(answer {LAYER_COUNT_KEY}=4 for a 4-layer board with ground / power planes)", layers.value, None))
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
    return Stackup(copper=copper, dielectrics=list(stackup.dielectrics), solder_mask=stackup.solder_mask, provenance=provenance or stackup.provenance)


def board_layers(stackup: Stackup) -> list[Layer]:
    """``PCBDesign.layers`` for the stack: a plane layer is KiCad type ``power``, every other one ``signal``."""
    return [Layer(name=c.name, kind="power" if c.plane_net is not None else "signal") for c in stackup.copper]


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


__all__ = [
    "GENERIC_NOTE",
    "GENERIC_STACKS",
    "PLANE_EDGE_CLEARANCE_MM",
    "STACKUP_TOOL",
    "STACKUP_VERSION",
    "GenericStack",
    "board_layers",
    "generic_stackup",
    "plane_description",
    "plane_zones",
    "stack_description",
    "stackup_choices",
    "with_planes",
]
