"""Board base materials and the rule that proposes one from the device's needs: FR-4, a high-frequency laminate, polyimide flex, an aluminium IMS.

Invariant: every number here was read from the manufacturer's document
opened for it on 2026-11-13 - title, revision, URL and page / table are in
the ``SRC_*`` constants and in every number's provenance note - or is a
stated choice of this module that says so. Nothing is estimated and nothing
comes from memory. The FR-4 entry is the generic stack of
:mod:`ai_eda.design.stackup`, unchanged ("a generic value, not a fab's": no
manufacturer's data sheet stands behind it), and a board whose requirements
name no material and no material need keeps it byte for byte - no table
row, no IR field, no hash changes (:attr:`MaterialDecision.untouched`).

The materials (:data:`MATERIALS`; the ids are the ``pcb_material`` answers):

* ``fr4`` - the generic FR-4 stacks (2 / 4 layers, er 4.5 at 1 MHz, a
  stated choice).
* ``rogers_ro4350b`` - Rogers RO4350B hydrocarbon / ceramic cores (design Dk
  3.66, Df 0.0037 at 10 GHz, :data:`SRC_RO4000` p. 3); the 4-layer stack
  bonds its two cores with one ply of RO4450F bondply (Dk 3.52 at 10 GHz,
  0.102 mm, :data:`SRC_RO4400`), which Rogers names compatible with RO4350B.
* ``polyimide_flex`` - DuPont Pyralux AP9121R, an adhesiveless all-polyimide
  double-sided laminate: 35 um rolled-annealed copper on a 50 um dielectric,
  Dk 3.4 / Df 0.002 at 1 MHz (:data:`SRC_PYRALUX_AP` Tables 2 and 3).
* ``aluminium_ims`` - Aismalibar COBRITHERM HTC 3,2W: 35 um copper on a
  90 um dielectric of 3.00 W/mK on a 1.5 mm aluminium base
  (:data:`SRC_COBRITHERM`); one copper layer.

Each material's generic stacks (:data:`MATERIAL_STACKS`: RO4350B 2 / 4
layers, flex 2 layers, IMS 1 copper layer) take only thicknesses the data
sheet offers (:func:`material_stack_problems`) and pass the same rule checks
as the generic stacks where they apply (:func:`~ai_eda.design.stackup.stack_problems`
for a stack with planes, :func:`~ai_eda.design.stackup.via_drill_problems`
for one with a through via).

The selection rule (:func:`select_material`, deterministic) reads the
confirmed requirements of a design category - exactly those the closed world
counted before this module (:func:`_stated`; a model's unconfirmed value or
an ``other`` item is not stated, so an IR that built before builds as then):

* ``pcb_material`` - the user's own choice wins; a stated need it
  contradicts (a bending board on a rigid material) refuses with the reason,
  a stated carrier at or above the threshold is only noted.
* ``carrier_frequency`` at or above :data:`HF_THRESHOLD_HZ` (500 MHz) proposes
  ``rogers_ro4350b``: Rogers states that the laminates available to
  designers narrow once operating frequencies reach 500 MHz and above
  (:data:`HF_RULE_SOURCE`). The threshold is this module's stated choice
  taken from that sentence. A carrier below it is not read at all (the KR
  447 MHz boards stay as they are).
* ``pcb_flexing`` = ``static`` (bent once to install) proposes
  ``polyimide_flex`` with the minimum bend radius 10 x the flex thickness of
  IPC-2223 Use A for a 2-layer flex area as Wurth Elektronik's flex guide
  states it (:data:`FLEX_RULE_SOURCE`); ``dynamic`` (bends in use) is refused:
  the guide sends dynamic bending to its specialists and says a dynamic flex
  ideally has one copper layer in the bend, while the generic flex stack has
  two - no sourced rule, no stack. ``none`` keeps FR-4.
* ``power_dissipation`` (W) above 0 never decides: no document read here
  says at which dissipation FR-4 stops being adequate, so the rule asks the
  required question ``pcb_material`` (IMS or FR-4) instead. So does a set of
  needs that propose two different materials.

A proposed material is a row of the ``confirm_design`` table (a choice the
user confirms; never confirmed here) and every number of its stack is
stamped like every other choice (:func:`~ai_eda.design.base.choice_provenance`).

Consequences stated, not hidden (each is a table line):

* The compiled ``.kicad_pcb`` carries no ``(stackup)`` section in this
  version: KiCad shows its own default stack, not the material, and the
  board file's thickness (and the 3D preview's) stays the default 1.6 mm
  unless ``ir.pcb.manufacturing.board_thickness_mm`` is the user's or a
  fab's - the fab order must name the material and the stack.
* Impedance and delay read the material's er from the stack like any stack
  (``calc.tline``); without a plane (2-layer, flex, IMS) impedance stays
  NOT_VERIFIED "impedance is undefined without a reference plane".
* IMS: one copper layer. A KiCad board has an even copper layer count, so
  the board keeps ``B.Cu`` and the PCB agent adds keep-out
  :data:`SINGLE_LAYER_KEEPOUT_ID` over the outline on ``B.Cu`` (tracks, vias,
  pads, zones; :func:`single_layer_keepout`) - the router's routing.maze 0.4
  keep-outs then route ``F.Cu`` only (maze.py is not changed), a through-hole
  part's pads break the keep-out (``pcb.keepout`` and DRC say so) and a net
  that needs a crossing stays unrouted. The dielectric's thermal
  conductivity is recorded; no check computes a temperature from it.
* Flex: the bend zone, the coverlay and "no vias / pads in the bend" are
  not in the IR and are not checked.

Everything is a pure function of its arguments (no clock, no randomness).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai_eda.ir import (
    BoardOutline,
    CircuitIR,
    Keepout,
    MetalBase,
    MissingInformation,
    Provenance,
    ProvenanceKind,
    Requirement,
    StackMaterial,
    Stackup,
    StackupCopper,
    StackupDielectric,
    Traced,
)
from ai_eda.ir.pcb import DielectricKind

from ai_eda.design.base import DESIGN_CATEGORIES, Choice, LayerPolicy, choice_provenance, leave_out_answer, unusable_remedy
from ai_eda.design.inputs import (
    FLEXING_ALIASES,
    FLEXING_KEY,
    LAYER_COUNT_KEY,
    MATERIAL_ALIASES,
    MATERIAL_KEY,
    POWER_DISSIPATION_ALIASES,
    POWER_DISSIPATION_KEY,
    LayerCountInput,
    read_inputs,
    read_value,
)
from ai_eda.design.stackup import (
    ESCALATION_TOOL,
    ESCALATION_VERSION,
    ICD_STACKUPS,
    PLANE_EDGE_CLEARANCE_MM,
    Direction,
    GenericStack,
    PlaneRole,
    layer_count_choice_text,
    plane_description,
    stack_problems,
    via_drill_problems,
)

#: provenance ``tool`` / ``tool_version`` of what this module derives (the IMS keep-out); bumped when a stack or the rule changes
MATERIALS_TOOL = "design.materials"
MATERIALS_VERSION = "0.1"

# --------------------------------------------------------------------------- the documents (each opened 2026-11-13)

#: Rogers' RO4000 laminate data sheet (RO4003C and RO4350B), served on rogerscorp.com
SRC_RO4000 = ("Rogers Corporation 'RO4000 Series High Frequency Circuit Materials' data sheet, Revised 1592 080322, PUB# 92-004 (c) 2022, "
              "https://www.rogerscorp.com/-/media/project/rogerscorp/documents/advanced-electronics-solutions/english/data-sheets/"
              "ro4000-laminates-ro4003c-and-ro4350b---data-sheet.pdf")
#: Rogers' RO4450F / RO4460G2 bondply data sheet, served on rogerscorp.com
SRC_RO4400 = ("Rogers Corporation 'RO4450F and RO4460G2 Bondply' data sheet, Issued 1595 090826, Publication #92-150 (c) 2022, "
              "https://www.rogerscorp.com/-/media/project/rogerscorp/documents/advanced-electronics-solutions/english/data-sheets/"
              "ro4400-series-bondply-data-sheet---ro4450f-and-ro4460g2-bondply.pdf")
#: DuPont's Pyralux AP data sheet as its distributor Insulectro serves it (dupont.com redirects Pyralux AP to qnityelectronics.com,
#: whose page lists no data sheet link)
SRC_PYRALUX_AP = ("DuPont 'Pyralux AP All-Polyimide Double-Sided Copper-Clad Laminate' data sheet EI-10124 (2/25), (c) 2025 DuPont, "
                  "https://insulectro.com/wp-content/uploads/2025/04/Pyralux-AP-Laminate-Data-Sheet-2-25.pdf")
#: Aismalibar's (Benmayor S.A.) COBRITHERM data sheet as Eurocircuits serves it
SRC_COBRITHERM = ("Aismalibar (Benmayor S.A.) 'COBRITHERM HTC 3,2W 90-130um' Data Sheet DS_21 Rev 01, "
                  "https://www.eurocircuits.com/content/uploads/2026/07/DS_21_COBRITHERM-HTC-32W-90µm–130µm.pdf")
#: Wurth Elektronik's flex design guide
SRC_WURTH_FLEX = ("Wurth Elektronik 'Flex Solutions Design Guide' Version 06.2025 / 504720 EN, "
                  "https://www.we-online.com/files/pdf1/design-guide-flex-solutions-cbt-en.pdf")

# --------------------------------------------------------------------------- the selection rule's thresholds (stated choices with sources)

#: the RF carrier at and above which the rule proposes the high-frequency laminate (Hz): this module's stated choice
HF_THRESHOLD_HZ = 500e6
#: where the 500 MHz comes from (paraphrased; the data sheet's own words are on its p. 1)
HF_RULE_SOURCE = (f"{SRC_RO4000} p. 1: the selection of laminates available to designers is significantly reduced once operational frequencies "
                  "increase to 500 MHz and above, and the low dielectric loss of RO4000 serves applications where higher operating frequencies "
                  "limit conventional circuit board laminates")
#: minimum bend radius / flex thickness of a 2-layer flex area bent to install (IPC-2223 Use A) as Wurth states it
FLEX_STATIC_BEND_RATIO = 10.0
FLEX_RULE_SOURCE = (f"{SRC_WURTH_FLEX} p. 15 table 'Design bending radii depending on flex thickness': IPC-2223 Use A (flex-to-install), "
                    "flex area 1 layer thickness x 10, 2 layers thickness x 10, 4 layers thickness x 20 (the IPC text itself was not opened)")
#: why a dynamically flexing board gets no stack in this version
DYNAMIC_FLEX_SOURCE = (f"{SRC_WURTH_FLEX} p. 18: for IPC-2223 Use B (dynamic bending load) consult Wurth's specialists; p. 4: dynamic flexes ideally have "
                       "only one layer in the bending area; p. 14: flexible solder mask is not for dynamic bending (coverlay is)")

#: the answers ``pcb_flexing`` takes: bent once to install, bends in use, does not bend
FLEXING_VALUES: tuple[str, ...] = ("static", "dynamic", "none")

#: the id of the keep-out that keeps copper off ``B.Cu`` of a one-copper-layer (IMS) board
SINGLE_LAYER_KEEPOUT_ID = "ims_bcu"

#: what every material stack's description ends with
MATERIAL_NOTE = ("typical values from the manufacturer's data sheet, not a specification and not a fab's stack (a fab's stack is not read into the IR "
                 "in this version; only the fab's measured impedance is real)")

MaterialKind = Literal["rigid", "flex", "ims"]


@dataclass(frozen=True)
class Material:
    """One base material: its answer id, product name, kind, the documents its numbers come from and the facts no calculator reads."""

    id: str
    name: str
    kind: MaterialKind
    sources: tuple[str, ...]
    facts: tuple[str, ...] = ()


#: the base materials, in answer order (``pcb_material=<id>``)
MATERIALS: dict[str, Material] = {
    "fr4": Material(
        id="fr4", name="FR-4 (generic)", kind="rigid",
        sources=("none: the generic FR-4 stacks of ai_eda.design.stackup are stated choices ('a generic value, not a fab's'), not a data sheet's",),
    ),
    "rogers_ro4350b": Material(
        id="rogers_ro4350b", name="Rogers RO4350B", kind="rigid", sources=(SRC_RO4000, SRC_RO4400),
        facts=(
            f"RO4350B Tg >280 C (TMA, IPC-TM-650 2.4.24.3), Td 390 C, Z-axis CTE 32 ppm/C (-55 to 288 C), process Dk 3.48 +/- 0.05 at 10 GHz / 23 C "
            f"(IPC-TM-650 2.5.5.5), Df 0.0031 at 2.5 GHz; the design Dk falls by about 0.1 as the core goes from 0.020 in to 0.004 in ({SRC_RO4000} p. 3)",
            f"RO4450F Tg >280 C (TMA), Df 0.004 at 10 GHz; compatible in multilayer constructions with RO4350B ({SRC_RO4400} pp. 1-2)",
        ),
    ),
    "polyimide_flex": Material(
        id="polyimide_flex", name="DuPont Pyralux AP (AP9121R)", kind="flex", sources=(SRC_PYRALUX_AP, SRC_WURTH_FLEX),
        facts=(
            f"AP9121 Tg 220 C (DuPont method, TMA), Dk 3.2 and Df 0.003 at 10 GHz (ASTM D2520), dielectric strength 200 V/um (ASTM D149), "
            f"flexural endurance 6,000 cycles (IPC-TM-650 2.4.3) - typical values ({SRC_PYRALUX_AP} Table 3); 'R' = rolled-annealed copper (Table 2 note)",
        ),
    ),
    "aluminium_ims": Material(
        id="aluminium_ims", name="Aismalibar COBRITHERM HTC 3,2W", kind="ims", sources=(SRC_COBRITHERM,),
        facts=(
            f"dielectric Tg 120 C (DSC, IPC-TM-650 2.4.25), maximum operating temperature 150 C, Td 422 C, AC breakdown >= 4 kV at 90 um "
            f"(IPC-TM-650 2.5.6.3, a material reference, not guaranteed), DC proof test 1500 V at 90 um, CTI 600 V; thermal conductivity may deviate "
            f"by up to +/- 15 %, and p. 1 describes the bonding layer as 3.2 W/mK where the property table gives 3.00 ({SRC_COBRITHERM} pp. 1-2)",
        ),
    ),
}
#: the ``pcb_material`` answers
MATERIAL_IDS: tuple[str, ...] = tuple(MATERIALS)
#: the material a board gets when nothing names one: the generic stack of :mod:`ai_eda.design.stackup`
DEFAULT_MATERIAL = "fr4"


@dataclass(frozen=True)
class MaterialDielectric:
    """One dielectric of a material stack, every number as its data sheet states it (``source``: where)."""

    kind: DielectricKind
    thickness_mm: float
    er: float
    #: the frequency the data sheet states ``er`` at (Hz); ``None`` when it states a band (``er_note`` says which)
    er_hz: float | None
    er_note: str
    loss_tangent: float
    df_hz: float
    product: str
    source: str
    thermal_conductivity_w_mk: float | None = None


@dataclass(frozen=True)
class MaterialStack:
    """A material's generic stack: copper ``(name, um)`` top to bottom, its dielectrics and, for an IMS, the metal base ``(metal, mm)``.

    ``planes`` / ``directions`` / ``roles_source`` mean what they mean in
    :class:`~ai_eda.design.stackup.GenericStack`; :meth:`generic` gives the
    stack in that form for the shared rule checks.
    """

    material: str
    title: str
    copper: tuple[tuple[str, float], ...]
    copper_source: str
    dielectrics: tuple[MaterialDielectric, ...]
    planes: tuple[tuple[str, PlaneRole], ...] = ()
    directions: tuple[tuple[str, Direction], ...] = ()
    roles_source: str = ""
    metal_base: tuple[str, float] | None = None
    metal_base_source: str = ""

    @property
    def layer_count(self) -> int:
        return len(self.copper)

    @property
    def thickness_mm(self) -> float:
        """Copper, dielectrics and the metal base: the whole board."""
        return sum(d.thickness_mm for d in self.dielectrics) + sum(um for _, um in self.copper) / 1000.0 + (self.metal_base[1] if self.metal_base else 0.0)

    def generic(self) -> GenericStack:
        """The stack as a :class:`~ai_eda.design.stackup.GenericStack` (kind, mm, er, Hz per dielectric) for ``stack_problems`` / ``via_drill_problems``."""
        return GenericStack(
            layer_count=self.layer_count, title=self.title, copper=self.copper,
            dielectrics=tuple((d.kind, d.thickness_mm, d.er, d.er_hz) for d in self.dielectrics),  # type: ignore[misc]
            planes=self.planes, directions=self.directions, roles_source=self.roles_source,
        )


# RO4350B (SRC_RO4000 p. 3 table / p. 4 standard thicknesses and claddings) and RO4450F (SRC_RO4400 p. 1 thickness, p. 2 table)
_RO4350B_DK = "design Dk, stated for 8 to 40 GHz (differential phase length method)"


def _ro4350b(mm: float) -> MaterialDielectric:
    return MaterialDielectric(
        kind=DielectricKind.CORE, thickness_mm=mm, er=3.66, er_hz=None, er_note=_RO4350B_DK, loss_tangent=0.0037, df_hz=10e9, product="Rogers RO4350B",
        thermal_conductivity_w_mk=0.69,
        source=(f"{SRC_RO4000} p. 3 table (design Dk 3.66, 8 to 40 GHz; Df 0.0037 at 10 GHz / 23 C; thermal conductivity 0.69 W/m/K at 80 C, ASTM C518) "
                "and p. 4 standard thicknesses"),
    )


_RO4450F = MaterialDielectric(
    kind=DielectricKind.PREPREG, thickness_mm=0.102, er=3.52, er_hz=10e9, er_note="", loss_tangent=0.004, df_hz=10e9, product="Rogers RO4450F",
    thermal_conductivity_w_mk=0.65,
    source=f"{SRC_RO4400} p. 1 standard thickness 0.0040 in (0.102 mm) and p. 2 table (Dk 3.52 +/- 0.05 and Df 0.004 at 10 GHz, IPC TM-650 2.5.5.5; 0.65 W/(m.K))",
)
_RO_COPPER = f"{SRC_RO4000} p. 4 standard claddings: 1/2 oz (18 um) HH/HH, 1 oz (35 um) H1/H1 electrodeposited copper"
_AP9121 = MaterialDielectric(
    kind=DielectricKind.CORE, thickness_mm=0.05, er=3.4, er_hz=1e6, er_note="", loss_tangent=0.002, df_hz=1e6, product="DuPont Pyralux AP (AP9121R)",
    source=f"{SRC_PYRALUX_AP} Table 2 (AP9121R: 35 um copper, 50 um dielectric) and Table 3 (AP9121 Dk 3.4 and Df 0.002 at 1 MHz, IPC-TM-650 2.5.5.3)",
)
_COBRITHERM = MaterialDielectric(
    kind=DielectricKind.CORE, thickness_mm=0.09, er=5.2, er_hz=10e3, er_note="after damp heat and recovery", loss_tangent=0.015, df_hz=10e3,
    product="Aismalibar COBRITHERM HTC 3,2W", thermal_conductivity_w_mk=3.00,
    source=(f"{SRC_COBRITHERM} p. 1 standard construction (isolation 90 / 130 um, +/- 15 um) and property table (relative permittivity 5.2 and "
            "dissipation factor 0.015 at 10 kHz after damp heat and recovery, IPC-TM-650 2.5.5.9; thermal conductivity 3.00 W/mK, ASTM D5470)"),
)
_ICD_4 = f"{ICD_STACKUPS} Fig. 2 (Top / GND / VCC / Bottom)"

#: every material's generic stacks by (material id, copper layer count) - each number from the data sheet its dielectric / copper names
MATERIAL_STACKS: dict[tuple[str, int], MaterialStack] = {
    ("rogers_ro4350b", 2): MaterialStack(
        material="rogers_ro4350b", title="2-layer Rogers RO4350B 1.59 mm", copper=(("F.Cu", 35.0), ("B.Cu", 35.0)), copper_source=_RO_COPPER,
        dielectrics=(_ro4350b(1.52),),
    ),
    ("rogers_ro4350b", 4): MaterialStack(
        material="rogers_ro4350b", title="4-layer Rogers RO4350B / RO4450F 1.262 mm",
        copper=(("F.Cu", 35.0), ("In1.Cu", 35.0), ("In2.Cu", 35.0), ("B.Cu", 35.0)), copper_source=_RO_COPPER,
        dielectrics=(_ro4350b(0.51), _RO4450F, _ro4350b(0.51)),
        planes=(("In1.Cu", "ground"), ("In2.Cu", "power")), directions=(("F.Cu", "horizontal"), ("B.Cu", "vertical")), roles_source=_ICD_4,
    ),
    ("polyimide_flex", 2): MaterialStack(
        material="polyimide_flex", title="2-layer polyimide flex (Pyralux AP9121R) 0.12 mm", copper=(("F.Cu", 35.0), ("B.Cu", 35.0)),
        copper_source=f"{SRC_PYRALUX_AP} Table 2 (AP9121R: 35 um rolled-annealed copper each side)", dielectrics=(_AP9121,),
    ),
    ("aluminium_ims", 1): MaterialStack(
        material="aluminium_ims", title="1-copper-layer aluminium IMS (COBRITHERM HTC 3,2W) 1.625 mm", copper=(("F.Cu", 35.0),),
        copper_source=f"{SRC_COBRITHERM} p. 1 standard construction (ED copper 35 / 70 / 105 um)", dielectrics=(_COBRITHERM,),
        metal_base=("aluminium 5052 / 6061", 1.5),
        metal_base_source=f"{SRC_COBRITHERM} p. 1 standard construction (aluminium 800 / 1000 / 1200 / 1500 / 2000 / 2500 um, alloy 5052 / 6061)",
    ),
}

#: what each data sheet offers (mm): only these thicknesses may appear in a material stack (:func:`material_stack_problems`)
OFFERED_DIELECTRIC_MM: dict[str, tuple[float, ...]] = {
    "Rogers RO4350B": (0.10, 0.17, 0.25, 0.51, 0.76, 1.52),        # SRC_RO4000 p. 4
    "Rogers RO4450F": (0.102,),                                     # SRC_RO4400 p. 1
    "DuPont Pyralux AP (AP9121R)": (0.05,),                         # SRC_PYRALUX_AP Table 2, AP9121R
    "Aismalibar COBRITHERM HTC 3,2W": (0.09, 0.13),                 # SRC_COBRITHERM p. 1
}
#: copper thicknesses (um) each material's data sheet offers
OFFERED_COPPER_UM: dict[str, tuple[float, ...]] = {
    "rogers_ro4350b": (18.0, 35.0),                                  # SRC_RO4000 p. 4
    "polyimide_flex": (35.0,),                                       # SRC_PYRALUX_AP Table 2, AP9121R
    "aluminium_ims": (35.0, 70.0, 105.0),                            # SRC_COBRITHERM p. 1
}
#: metal base thicknesses (mm) the IMS data sheet offers
OFFERED_BASE_MM: tuple[float, ...] = (0.8, 1.0, 1.2, 1.5, 2.0, 2.5)  # SRC_COBRITHERM p. 1


def material_stack(material: str, layer_count: int) -> MaterialStack:
    """The generic stack of ``material`` with ``layer_count`` copper layers; ``ValueError`` naming the counts it has."""
    if (material, layer_count) not in MATERIAL_STACKS:
        counts = sorted(n for m, n in MATERIAL_STACKS if m == material)
        raise ValueError(f"no generic {material} stack with {layer_count} copper layer(s) (it has {counts or 'none'})")
    return MATERIAL_STACKS[(material, layer_count)]


def material_stack_problems(ms: MaterialStack, drill_mm: float = 0.4) -> list[str]:
    """Every rule ``ms`` breaks (empty: none): only offered thicknesses, the shape its kind needs, and the generic stacks' rules where they apply.

    * every dielectric thickness is one its data sheet offers
      (:data:`OFFERED_DIELECTRIC_MM`), every copper thickness one the
      material's data sheet offers (:data:`OFFERED_COPPER_UM`), an IMS base one
      of :data:`OFFERED_BASE_MM`;
    * an IMS stack has one copper layer and its metal base, any other stack
      none and an even copper count; a flex stack two copper layers (the one
      generic flex build);
    * a stack with planes passes :func:`~ai_eda.design.stackup.stack_problems`,
      a stack with a through via :func:`~ai_eda.design.stackup.via_drill_problems`
      at ``drill_mm`` (the router's default via drill, 0.4 mm).
    """
    out: list[str] = []
    m = MATERIALS.get(ms.material)
    if m is None:
        return [f"{ms.title}: material {ms.material!r} is not one of {list(MATERIAL_IDS)}"]
    if len(ms.dielectrics) != (1 if m.kind == "ims" else ms.layer_count - 1):
        out.append(f"{ms.title}: {ms.layer_count} copper layer(s) with {len(ms.dielectrics)} dielectric(s)")
    for i, d in enumerate(ms.dielectrics):
        offered = OFFERED_DIELECTRIC_MM.get(d.product, ())
        if d.thickness_mm not in offered:
            out.append(f"{ms.title}: dielectric {i} {d.product} {d.thickness_mm:g} mm is not a thickness its data sheet offers {list(offered)}")
    copper = OFFERED_COPPER_UM.get(ms.material, ())
    out += [f"{ms.title}: {name} {um:g} um is not a copper thickness the data sheet offers {list(copper)}" for name, um in ms.copper if um not in copper]
    if m.kind == "ims":
        if ms.layer_count != 1 or ms.metal_base is None:
            out.append(f"{ms.title}: an IMS stack in this version has exactly one copper layer and its metal base")
        elif ms.metal_base[1] not in OFFERED_BASE_MM:
            out.append(f"{ms.title}: metal base {ms.metal_base[1]:g} mm is not a thickness the data sheet offers {list(OFFERED_BASE_MM)}")
        return out
    if ms.metal_base is not None:
        out.append(f"{ms.title}: only an IMS stack has a metal base")
    if ms.layer_count < 2 or ms.layer_count % 2:
        out.append(f"{ms.title}: {ms.layer_count} copper layers - a KiCad board has an even count >= 2")
    if m.kind == "flex" and ms.layer_count != 2:
        out.append(f"{ms.title}: the generic flex stack has two copper layers")
    g = ms.generic()
    if ms.planes:
        out += stack_problems(g)
    out += via_drill_problems(g, drill_mm)
    return out


# --------------------------------------------------------------------------- reading the requirements


def _stated(ir: CircuitIR, aliases: tuple[str, ...]) -> list[Requirement]:
    """The requirements under ``aliases`` the rule reads: a design category (:data:`~ai_eda.design.base.DESIGN_CATEGORIES`) whose value
    is not a model's or an assumption awaiting confirmation (``needs_verification``).

    Exactly the requirements the closed world (:func:`~ai_eda.design.base.unserved_requirements`)
    counts, and counted before this module (no template served these keys, so
    every one of them refused the template): any other - a model's implicit
    item a confirmation leaves unconfirmed, a confirmed item filed under
    ``other`` or ``manufacturing`` - never stopped a design and is not stated
    here either, so every IR that built before builds exactly as then
    (:attr:`MaterialDecision.untouched`).
    """
    return [r for r in ir.requirements.requirements if r.key in aliases and r.category in DESIGN_CATEGORIES
            and not (r.value is not None and r.value.provenance.needs_verification)]


def _categorical(ir: CircuitIR, aliases: tuple[str, ...], allowed: tuple[str, ...]) -> tuple[tuple[Requirement, str] | None, str | None]:
    """``((requirement, value), None)`` when the confirmed requirements under ``aliases`` state exactly one of ``allowed``; ``(None, why)``; ``(None, None)`` when none states it.

    The whole value must be one allowed word (case-insensitive, surrounding
    whitespace ignored, ``-`` read as ``_``); a description is never mapped to
    one, and requirements that state different values are ambiguous. Only
    the requirements :func:`_stated` reads count.
    """
    candidates = _stated(ir, aliases)
    if not candidates:
        return None, None
    found: list[tuple[Requirement, str]] = []
    reasons: list[str] = []
    for r in candidates:
        if r.value is None:
            reasons.append(f"{r.id}: has no value")
            continue
        if not r.value.provenance.is_authoritative:
            reasons.append(f"{r.id}: value is {r.value.provenance.kind.value}, not yet the user's (confirm it, or answer {r.key} directly)")
            continue
        raw = r.value.value
        word = raw.strip().lower().replace("-", "_") if isinstance(raw, str) else None
        if word not in allowed:
            reasons.append(f"{r.id}: {raw!r} is not exactly one of {', '.join(allowed)}")
            continue
        found.append((r, word))
    if reasons:
        return None, "; ".join(reasons)
    if len({w for _, w in found}) > 1:
        return None, "ambiguous: " + ", ".join(f"{r.id} says {w}" for r, w in found)
    return found[0], None


def read_material_answer(ir: CircuitIR) -> tuple[tuple[Requirement, str] | None, str | None]:
    """The user's ``pcb_material`` (one of :data:`MATERIAL_IDS`), ``(None, why)`` when unusable, ``(None, None)`` when not stated."""
    return _categorical(ir, MATERIAL_ALIASES, MATERIAL_IDS)


def read_flexing(ir: CircuitIR) -> tuple[tuple[Requirement, str] | None, str | None]:
    """The confirmed ``pcb_flexing`` (one of :data:`FLEXING_VALUES`), ``(None, why)`` when unusable, ``(None, None)`` when not stated."""
    return _categorical(ir, FLEXING_ALIASES, FLEXING_VALUES)


def read_power_dissipation(ir: CircuitIR) -> tuple[tuple[Requirement, Traced] | None, str | None]:
    """The confirmed ``power_dissipation`` in W (>= 0, one number), ``(None, why)`` when unusable, ``(None, None)`` when not stated
    (only the requirements :func:`_stated` reads count)."""
    candidates = _stated(ir, POWER_DISSIPATION_ALIASES)
    if not candidates:
        return None, None
    found: list[tuple[Requirement, Traced]] = []
    reasons: list[str] = []
    for r in candidates:
        traced, why = read_value(r, "W")
        if traced is None:
            reasons.append(why or f"{r.id}: unreadable")
        elif float(traced.value) < 0.0:
            reasons.append(f"{r.id}: {float(traced.value):.12g} W is not a dissipation (below 0 W)")
        else:
            found.append((r, traced))
    if reasons:
        return None, "; ".join(reasons)
    if len({float(t.value) for _, t in found}) > 1:
        return None, "ambiguous: " + ", ".join(f"{r.id} says {float(t.value):.12g} W" for r, t in found)
    return found[0], None


# --------------------------------------------------------------------------- the selection rule


@dataclass(frozen=True)
class MaterialDecision:
    """What :func:`select_material` decided: a material (and why), or a question, or a refusal.

    ``read`` are the requirements the decision rests on - the stack built
    from it serves them (its provenance names them). ``proposed`` is true
    when the rule proposed the material (a ``confirm_design`` row the user
    confirms) and false when the user answered it or nothing points away from
    FR-4. ``notes`` are table lines (a need the user's answer overrides).
    """

    material: str = DEFAULT_MATERIAL
    reason: str = ""
    read: tuple[Requirement, ...] = ()
    stated: Requirement | None = None
    proposed: bool = False
    question: MissingInformation | None = None
    refusal: str | None = None
    notes: tuple[str, ...] = ()

    @property
    def untouched(self) -> bool:
        """Nothing about the material was stated or needed: the generic FR-4 path, byte for byte as before this module."""
        return self.material == DEFAULT_MATERIAL and not self.read and self.question is None and self.refusal is None


def _name(material: str) -> str:
    return MATERIALS[material].name


def _ims_offer() -> str:
    d, base = _COBRITHERM, MATERIAL_STACKS[("aluminium_ims", 1)].metal_base
    assert base is not None
    return (f"aluminium_ims ({_name('aluminium_ims')}: a {d.thickness_mm * 1000:g} um dielectric of {d.thermal_conductivity_w_mk:.2f} W/mK on a {base[1]:g} mm "
            f"aluminium base, designed for carrying the parts' heat to the base - {SRC_COBRITHERM} p. 1; one copper layer: B.Cu is kept out, a through-hole "
            "part's pads break that keep-out and a net that needs a crossing stays unrouted)")


def _question(text: str, rationale: str) -> MissingInformation:
    return MissingInformation(key=MATERIAL_KEY, required=True, question=text, rationale=rationale, options=list(MATERIAL_IDS))


def select_material(ir: CircuitIR) -> MaterialDecision:
    """The board's base material from the confirmed requirements (module docstring): a material, a required question or a refusal.

    Deterministic, and a closed set of rules: the user's ``pcb_material``;
    else a static ``pcb_flexing`` proposes ``polyimide_flex``, a
    ``carrier_frequency`` at or above :data:`HF_THRESHOLD_HZ` proposes
    ``rogers_ro4350b``; a ``power_dissipation`` above 0 W, or two needs
    proposing different materials, ask; a dynamic ``pcb_flexing`` refuses;
    nothing stated keeps FR-4 (:attr:`MaterialDecision.untouched`). An
    unreadable or ambiguous material key refuses with the answer that clears
    it.
    """
    answer, why_a = read_material_answer(ir)
    flex, why_f = read_flexing(ir)
    power, why_p = read_power_dissipation(ir)
    reasons = [f"{key}: {why}" + (f"; {remedy}" if (remedy := unusable_remedy(ir, aliases)) else "")
               for key, why, aliases in ((MATERIAL_KEY, why_a, MATERIAL_ALIASES), (FLEXING_KEY, why_f, FLEXING_ALIASES),
                                         (POWER_DISSIPATION_KEY, why_p, POWER_DISSIPATION_ALIASES)) if why is not None]
    if reasons:
        return MaterialDecision(refusal="; ".join(reasons))
    inputs, _unusable = read_inputs(ir)
    carrier = inputs.get("carrier_frequency")
    hf = carrier if carrier is not None and float(carrier.traced.value) >= HF_THRESHOLD_HZ else None
    hf_text = (f"carrier_frequency {float(hf.traced.value) / 1e6:.12g} MHz ({hf.requirement.id}) is at or above the {HF_THRESHOLD_HZ / 1e6:g} MHz this rule takes "
               f"from {HF_RULE_SOURCE}") if hf is not None else ""
    read: list[Requirement] = [x[0] for x in (answer, flex, power) if x is not None]
    bending = flex[1] if flex is not None else None
    if bending == "dynamic":
        return MaterialDecision(refusal=(
            f"{FLEXING_KEY}: {flex[0].id} says the board bends in use (dynamic), and no bend rule for dynamic flexing is read in this version - "  # type: ignore[index]
            f"{DYNAMIC_FLEX_SOURCE} - while the generic polyimide flex stack has two copper layers; nothing is proposed. Answer {FLEXING_KEY}=static if the "
            f"board is only bent to install (leave this one out first: {leave_out_answer([flex[0].key])}), or provide the stack in the IR yourself"))  # type: ignore[index]
    if answer is not None:
        req, chosen = answer
        notes: list[str] = []
        if bending == "static" and chosen != "polyimide_flex":
            return MaterialDecision(refusal=(
                f"{MATERIAL_KEY}: {flex[0].id} says the board is bent to install, but {req.id} chooses {_name(chosen)}, a rigid board; "  # type: ignore[index]
                f"answer {MATERIAL_KEY}=polyimide_flex, or leave one of them out ({leave_out_answer([req.key])} or {leave_out_answer([flex[0].key])})"))  # type: ignore[index]
        if hf is not None and chosen != "rogers_ro4350b":
            notes.append(f"material rule: {hf_text} - it would propose rogers_ro4350b; your {req.id} = {chosen} is kept (the rule only proposes)")
        if power is not None:
            notes.append(f"material rule: {power[0].id} = {float(power[1].value):.12g} W is answered by your {req.id} = {chosen}; no check in this version "
                         "computes the board's temperature from the material")
        return MaterialDecision(material=chosen, reason=f"chosen by you: {req.id} = {chosen}", read=tuple(read), stated=req, notes=tuple(notes))
    proposals: list[tuple[str, str]] = []
    if bending == "static":
        proposals.append(("polyimide_flex", (
            f"{flex[0].id}: the board is bent to install (static); the generic 1.6 mm FR-4 stack is none of the bendable builds {SRC_WURTH_FLEX} names "  # type: ignore[index]
            f"(polyimide flex, pp. 2 and 10: a 50 um polyimide core standard; thin depth-milled FR-4 SEMI.flex, p. 7, has no stack in this version), and "
            f"its p. 15 gives a 2-layer flex area bent to install (IPC-2223 Use A) a minimum bend radius of {FLEX_STATIC_BEND_RATIO:g} x its thickness")))
    if hf is not None:
        proposals.append(("rogers_ro4350b", hf_text))
    heat = power is not None and float(power[1].value) > 0.0
    if heat or len(proposals) > 1:
        lines = [f"{power[0].id}: power_dissipation = {float(power[1].value):.12g} W is stated"] if heat else []  # type: ignore[index]
        lines += [f"{why} - it proposes {m}" for m, why in proposals]
        offers = [_ims_offer()] if heat else []
        offers += [f"{m} ({_name(m)})" for m, _ in proposals]
        offers.append("fr4 (the generic FR-4 stack, as before)")
        why_ask = ("no document read in this version says at which dissipation FR-4 stops being adequate (the generic FR-4 stack states no thermal "
                   "conductivity), so the material is not decided for you" if heat else "")
        if len(proposals) > 1:
            why_ask = "; ".join(x for x in (why_ask, "these needs propose different materials and no generic stack here serves them together") if x)
        needs = [k for k, on in ((power[0].key if power is not None else "", heat), (flex[0].key if flex is not None else "", bending == "static")) if on]
        text = (f"Board base material: {'; '.join(lines)}. {why_ask[0].upper() + why_ask[1:]}. Answer {MATERIAL_KEY}=<one of {', '.join(MATERIAL_IDS)}>: "
                f"{'; '.join(offers)}" + (f"; or leave the need out ({leave_out_answer(needs)})" if needs else "") + ".")
        return MaterialDecision(read=tuple(read), question=_question(text, f"{MATERIAL_KEY}: {why_ask}"))
    if proposals:
        chosen, why = proposals[0]
        used = [*read, *([hf.requirement] if chosen == "rogers_ro4350b" and hf is not None else [])]
        return MaterialDecision(material=chosen, reason=f"proposed by the material rule: {why}", read=tuple(used), proposed=True)
    said = [f"{flex[0].id}: the board does not bend"] if flex is not None else []  # type: ignore[index]
    said += [f"{power[0].id}: 0 W to dissipate"] if power is not None else []  # type: ignore[index]
    return MaterialDecision(reason="; ".join(said) + " - the generic FR-4 stack" if said else "", read=tuple(read))


# --------------------------------------------------------------------------- the board: layer count, the IR stack, the table rows


def material_layer_count(decision: MaterialDecision, layers: LayerCountInput, policy: LayerPolicy) -> tuple[int | None, str | None]:
    """``(copper layer count, None)`` of the decided material's stack under the stated / default ``layers`` and the template's ``policy``, or ``(None, why)``.

    FR-4 and the high-frequency laminate build the count ``layers`` already
    holds (2 or 4, the policy applied). Flex builds 2: a stated other count
    refuses, and so does a policy that does not allow 2 (its reason quoted).
    IMS builds 1 copper layer: any stated ``pcb_layers`` refuses (it counts
    copper layers, and the IMS stack has one), and so does a policy that does
    not allow 2 (a template whose lines need a plane).
    """
    m = MATERIALS[decision.material]
    why_policy = f": {policy.reason.strip()}" if policy.reason.strip() else ""
    if m.kind == "flex":
        if not layers.is_default and layers.value != 2:
            return None, f"{LAYER_COUNT_KEY}: {layers.value} layers stated, but the generic {m.name} stack has two copper layers (the only flex stack)"
        if 2 not in policy.allowed:
            return None, f"{MATERIAL_KEY}: the generic {m.name} stack has two copper layers, a count this template does not build ({policy.allowed_text()}){why_policy}"
        return 2, None
    if m.kind == "ims":
        if not layers.is_default:
            return None, (f"{LAYER_COUNT_KEY}: {layers.value} layers stated, but the generic {m.name} stack has one copper layer ({LAYER_COUNT_KEY} counts copper "
                          f"layers); leave it out ({leave_out_answer([layers.requirement.key])}) to build the IMS board")  # type: ignore[union-attr]
        if 2 not in policy.allowed:
            return None, f"{MATERIAL_KEY}: the generic {m.name} stack has one copper layer and no plane, which this template does not build ({policy.allowed_text()} layers){why_policy}"
        return 1, None
    if decision.material != DEFAULT_MATERIAL and (decision.material, layers.value) not in MATERIAL_STACKS:
        return None, f"{MATERIAL_KEY}: no generic {m.name} stack with {layers.value} layers"
    return layers.value, None


def _hz(f: float) -> str:
    """``10 kHz`` / ``1 MHz`` / ``10 GHz``: a stated test frequency as the data sheet writes it."""
    for scale, unit in ((1e9, "GHz"), (1e6, "MHz"), (1e3, "kHz")):
        if f >= scale:
            return f"{f / scale:g} {unit}"
    return f"{f:g} Hz"


def material_stack_description(ms: MaterialStack) -> str:
    """One line naming every number of a material stack and where each comes from (the ``stackup`` choice text)."""
    copper = ", ".join(f"{name} {um:g} um" for name, um in ms.copper)
    diel = "; ".join(
        f"{d.kind.value} {d.thickness_mm:g} mm {d.product} er {d.er:g} " + (f"@ {_hz(d.er_hz)}" if d.er_hz is not None else f"({d.er_note})")
        + (f" ({d.er_note})" if d.er_hz is not None and d.er_note else "") + f", Df {d.loss_tangent:g} @ {_hz(d.df_hz)}"
        + (f", {d.thermal_conductivity_w_mk:.2f} W/mK" if d.thermal_conductivity_w_mk is not None else "") + f" [{d.source}]"
        for d in ms.dielectrics
    )
    base = f"; metal base {ms.metal_base[0]} {ms.metal_base[1]:g} mm [{ms.metal_base_source}]" if ms.metal_base is not None else ""
    return f"{ms.title}: copper {copper} [{ms.copper_source}]; dielectrics {diel}{base} - {MATERIAL_NOTE}"


def material_stackup(ms: MaterialStack, decision: MaterialDecision, template_id: str, *, confirmed: bool, ground_net: str = "GND",
                     power_net: str = "+5V") -> Stackup:
    """The IR :class:`~ai_eda.ir.Stackup` of a material stack, every number stamped as ``template_id``'s choice with its source in the note.

    Planes (a stack with plane roles) carry ``ground_net`` / ``power_net``
    like the generic 4-layer stack; ``material`` records the decision
    (:class:`~ai_eda.ir.StackMaterial`), ``metal_base`` the IMS plate. The
    stack-level provenance names the requirements the decision read
    (``derived_from``), so the stack serves them.
    """
    return _material_stackup(ms, decision.reason, lambda note: choice_provenance(template_id, note, confirmed), ground_net, power_net,
                             choice_provenance(template_id, material_stack_description(ms), confirmed))


def _material_stackup(ms: MaterialStack, reason: str, stamp, ground_net: str, power_net: str, provenance: Provenance) -> Stackup:
    """The IR stack of ``ms``: every number and plane net traced with ``stamp(note)`` (its source in the note), the material record with ``reason``."""
    m = MATERIALS[ms.material]
    title = ms.title

    def num(value: float, unit: str | None, note: str) -> Traced:
        return Traced(value=value, unit=unit, provenance=stamp(note))

    planes = {layer: (ground_net if role == "ground" else power_net) for layer, role in ms.planes}
    plane_text = plane_description(ground_net, power_net)
    copper = [
        StackupCopper(
            name=name, thickness_um=num(um, "um", f"{title}: {name} copper {um:g} um - {ms.copper_source}"),
            plane_net=Traced(value=planes[name], provenance=stamp(plane_text)) if name in planes else None,
        )
        for name, um in ms.copper
    ]
    dielectrics = []
    for i, d in enumerate(ms.dielectrics):
        what = f"{title}: dielectric {i} {d.product}"
        er_how = f"at {_hz(d.er_hz)}" if d.er_hz is not None else f"{d.er_note}; no single frequency is recorded"
        if d.er_hz is not None and d.er_note:
            er_how += f", {d.er_note}"
        dielectrics.append(StackupDielectric(
            kind=d.kind,
            thickness_mm=num(d.thickness_mm, "mm", f"{what} thickness {d.thickness_mm:g} mm - {d.source}"),
            er=num(d.er, None, f"{what} er {d.er:g} ({er_how}) - {d.source}"),
            er_frequency_hz=num(d.er_hz, "Hz", f"{what}: the frequency its er is stated at - {d.source}") if d.er_hz is not None else None,
            loss_tangent=num(d.loss_tangent, None, f"{what} Df {d.loss_tangent:g} at {_hz(d.df_hz)} - {d.source}"),
            material=d.product,
            thermal_conductivity_w_mk=(num(d.thermal_conductivity_w_mk, "W/mK", f"{what} thermal conductivity {d.thermal_conductivity_w_mk:g} W/mK - {d.source}")
                                       if d.thermal_conductivity_w_mk is not None else None),
        ))
    base = None
    if ms.metal_base is not None:
        base = MetalBase(material=ms.metal_base[0], thickness_mm=num(ms.metal_base[1], "mm", f"{title}: metal base {ms.metal_base[1]:g} mm - {ms.metal_base_source}"))
    ratio = None
    if m.kind == "flex":
        ratio = num(FLEX_STATIC_BEND_RATIO, None, f"{title}: minimum bend radius / flex thickness {FLEX_STATIC_BEND_RATIO:g} - {FLEX_RULE_SOURCE}")
    record = StackMaterial(id=m.id, name=m.name, kind=m.kind, reason=reason, sources=list(m.sources), facts=list(m.facts), min_bend_ratio=ratio)
    return Stackup(copper=copper, dielectrics=dielectrics, provenance=provenance, material=record, metal_base=base)


def material_escalation_stackup(ms: MaterialStack, reason: str, *, ground_net: str, power_net: str | None = None) -> Stackup:
    """A material's own generic stack chosen by the layer escalation (the PCB agent): :func:`escalation_stackup`'s counterpart for ``ms``.

    Every number keeps its data-sheet source in its note and is an
    ``assumption`` "chosen by the project owner's layer-escalation rule, not
    shown in a confirm_design table"; the ground planes carry ``ground_net``,
    the power planes ``power_net`` (the ground net when it is ``None``);
    ``reason`` is the board's material decision, carried over. The stack-level
    provenance is a placeholder the PCB agent replaces with the escalation
    record. ``ValueError`` for a stack that breaks
    :func:`material_stack_problems`.
    """
    problems = material_stack_problems(ms)
    if problems:
        raise ValueError(f"the {ms.title} stack breaks its rules: " + "; ".join(problems))
    power = power_net if power_net is not None else ground_net

    def stamp(note: str) -> Provenance:
        return Provenance(kind=ProvenanceKind.ASSUMPTION, note=f"{note} - chosen by the project owner's layer-escalation rule, not shown in a confirm_design table")

    placeholder = Provenance(kind=ProvenanceKind.ASSUMPTION, tool=ESCALATION_TOOL, tool_version=ESCALATION_VERSION,
                             note=f"layer count {ms.layer_count} ({ms.title}): a placeholder - the PCB agent replaces it with the layer-escalation record")
    return _material_stackup(ms, reason, stamp, ground_net, power, placeholder)


def material_rows(ms: MaterialStack, decision: MaterialDecision, layers: LayerCountInput, policy: LayerPolicy, *, ground_net: str, power_net: str,
                  edge_clearance_mm: float = PLANE_EDGE_CLEARANCE_MM) -> list[Choice]:
    """The ``confirm_design`` rows of a material stack: the proposed material, the layer count when it is a choice, the stack and its consequences."""
    m = MATERIALS[ms.material]
    out: list[Choice] = []
    if decision.proposed:
        out.append(Choice(MATERIAL_KEY, f"board base material: {m.name} - {decision.reason}; answer {MATERIAL_KEY}=<{' | '.join(MATERIAL_IDS)}> to choose another",
                          m.id, None))
    if layers.is_default:
        if m.kind == "rigid":
            out.append(Choice(LAYER_COUNT_KEY, layer_count_choice_text(policy), layers.value, None))
        elif m.kind == "flex":
            why = f" (this template's default is {policy.default}" + (f": {policy.reason.strip()})" if policy.reason.strip() else ")") if policy.default != 2 else ""
            out.append(Choice(LAYER_COUNT_KEY, f"board layer count: 2 layers, the only count of the generic {m.name} stack{why}", 2, None))
    out.append(Choice("stackup", material_stack_description(ms)))
    if ms.planes:
        out.append(Choice("stackup.planes", plane_description(ground_net, power_net)))
        out.append(Choice("stackup.plane_edge_clearance", "plane zones end this far inside the board edge", edge_clearance_mm, "mm"))
    if m.kind == "flex":
        t = sum(d.thickness_mm for d in ms.dielectrics) + sum(um for _, um in ms.copper) / 1000.0
        out.append(Choice("stackup.min_bend_radius", (
            f"minimum bend radius {FLEX_STATIC_BEND_RATIO:g} x the flex thickness ({FLEX_RULE_SOURCE}): at least {FLEX_STATIC_BEND_RATIO * t:.4g} mm over the "
            f"{t:.4g} mm clad laminate - the coverlay ({SRC_WURTH_FLEX} p. 14: polyimide typically 25 um with 25 or 50 um adhesive) is not in this stack and adds "
            f"to the thickness, so the real minimum is larger; the bend zone is not in the IR, so 'no vias and pads in the bending area' (p. 18) is not checked")))
    if m.kind == "ims":
        out.append(Choice("stackup.copper_layers", (
            f"one copper layer (F.Cu) on the aluminium base: a KiCad board has an even copper layer count, so the board keeps B.Cu and the PCB agent adds "
            f"keep-out {SINGLE_LAYER_KEEPOUT_ID} over the outline on B.Cu (tracks, vias, pads, zones) - the router then routes F.Cu only (a net that needs a "
            f"crossing stays unrouted) and a part with pads on B.Cu (a through-hole part) breaks the keep-out, which pcb.keepout and DRC report"), 1, None))
    return out


def material_board_lines(ms: MaterialStack, decision: MaterialDecision, source: str, ground_net: str, power_net: str) -> list[str]:
    """The table's board lines for a material stack: where it came from, the planes or why impedance is undefined, the facts and what is not done."""
    m = MATERIALS[ms.material]
    lines = [f"base material: {m.name} ({decision.reason})", f"stackup: {ms.title} ({source}); "
             + (f"planes In1.Cu = {ground_net}, In2.Cu = {power_net}" if ms.planes
                else f"no plane: impedance is undefined, only delay bounds are computed (er_max {max(d.er for d in ms.dielectrics):g})")]
    lines += [f"material fact (no check reads it): {f}" for f in m.facts]
    if m.kind == "ims":
        d = ms.dielectrics[0]
        lines.append(f"thermal: the dielectric's {d.thermal_conductivity_w_mk:.2f} W/mK is recorded in the stack; no check in this version computes a temperature "
                     "from it (domain.power.thermal judges components only)")
    lines.append(f"KiCad: the compiled .kicad_pcb carries no (stackup) section in this version, so KiCad shows its own default stack, not {m.name}, and "
                 f"its board thickness (and the 3D preview's) is the default 1.6 mm, not this stack's {ms.thickness_mm:.4g} mm (only a board_thickness_mm "
                 "in ir.pcb.manufacturing that is yours or a fab's is written); the fab order must name the material and the stack (not verified)")
    lines += list(decision.notes)
    return lines


# --------------------------------------------------------------------------- the IMS board: B.Cu kept out


def single_layer_keepout(stackup: Stackup | None, outline: BoardOutline | None) -> Keepout | None:
    """The keep-out that keeps all copper off ``B.Cu`` of a board whose stack has one copper layer on a metal base; ``None`` for every other board.

    The rectangle is the outline itself; it forbids tracks, vias (a through
    via is on every copper layer), pads and zones. Its provenance is
    ``derived`` by :data:`MATERIALS_TOOL` from the stack's metal base and the
    outline (it is no choice: it follows from the stack the user confirmed).
    """
    if stackup is None or stackup.metal_base is None or outline is None:
        return None
    rect = [float(outline.origin_x_mm), float(outline.origin_y_mm), float(outline.width_mm), float(outline.height_mm)]
    name = stackup.material.name if stackup.material is not None else stackup.metal_base.material
    prov = Provenance(kind=ProvenanceKind.DERIVED, tool=MATERIALS_TOOL, tool_version=MATERIALS_VERSION, derived_from=["pcb.stackup.metal_base", "pcb.outline"],
                      note=f"the stack ({name}) has one copper layer, F.Cu, on a metal base: B.Cu of the KiCad board holds no copper, over the whole outline")
    return Keepout(
        id=SINGLE_LAYER_KEEPOUT_ID, layers=["B.Cu"], rect=Traced(value=rect, unit="mm", provenance=prov), forbids=["tracks", "vias", "pads", "zones"],
        reason=f"one copper layer on a metal base ({name}): B.Cu holds no copper", provenance=prov,
    )


__all__ = [
    "DEFAULT_MATERIAL",
    "DYNAMIC_FLEX_SOURCE",
    "FLEXING_VALUES",
    "FLEX_RULE_SOURCE",
    "FLEX_STATIC_BEND_RATIO",
    "HF_RULE_SOURCE",
    "HF_THRESHOLD_HZ",
    "MATERIALS",
    "MATERIALS_TOOL",
    "MATERIALS_VERSION",
    "MATERIAL_IDS",
    "MATERIAL_NOTE",
    "MATERIAL_STACKS",
    "OFFERED_BASE_MM",
    "OFFERED_COPPER_UM",
    "OFFERED_DIELECTRIC_MM",
    "SINGLE_LAYER_KEEPOUT_ID",
    "SRC_COBRITHERM",
    "SRC_PYRALUX_AP",
    "SRC_RO4000",
    "SRC_RO4400",
    "SRC_WURTH_FLEX",
    "Material",
    "MaterialDecision",
    "MaterialDielectric",
    "MaterialStack",
    "material_board_lines",
    "material_escalation_stackup",
    "material_layer_count",
    "material_rows",
    "material_stack",
    "material_stack_description",
    "material_stack_problems",
    "material_stackup",
    "read_flexing",
    "read_material_answer",
    "read_power_dissipation",
    "select_material",
    "single_layer_keepout",
]
