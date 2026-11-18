"""The board's start and floor copper layer counts from its IC packages and parts (``design.package_layers`` 0.1, the owner's rule of 2026-11-13).

The project owner (2026-11-13): "IC 패키지에 따라서 통상적으로 사용하는 레이어 수가 정해져있잖아. 해당 값을 우선으로 조사해서 적용하고
레이어를 빼거나 추가하도록" - start from the layer count commonly used for the board's IC packages (researched, sourced), then add
layers when routing fails or remove layers while the board still routes (:data:`PACKAGE_RULE`). The PCB agent's layer escalation
(:mod:`ai_eda.agents.pcb`) does the adding and removing; this module only says where to start and how far down removal may go. It is
pure and deterministic: footprint facts, part identities and requirement values in, numbers and reasons out - no clock, no randomness,
no file but the KiCad footprint the caller already resolved.

The table (:data:`ROWS`) is the research synthesis ``package_layers.json`` (2026-11-13) row for row, every text field verbatim: the
match, ``start`` (the convention the search starts from), ``floor`` (the manufacturer's recommendation / minimum; 0 = none), the
source (title, revision, URL, section), the quote and the conditions. Only rows whose source was opened are kept (``opened`` is true
for every row, checked at import). A quote that came through an HTML extractor is kept verbatim as given and marked
(``quote_checked=False``, a comment above the row): its wording was not re-checked against the PDF.

Definitions (the synthesis' ``rules``; all from KiCad footprint facts, so the result is deterministic):

* **P** (pitch) is the ``P<x>mm`` token of the footprint name; without one, the smallest pad-centre spacing along x or y
  (:func:`pad_spacing`). **BODY** is the ``<w>x<h>mm`` token. **N** is the number of numbered copper pads (distinct numbers).
* A **BGA footprint** is one in library :data:`BGA_LIBRARY`, or one whose name holds a :data:`BGA_TOKENS` token.
* **D** (ball depth, :func:`ball_depth`): each numbered pad centre snapped to the P grid as column ``i`` / row ``j`` counted from the
  smallest x / y; ``ring = min(i, j, ncols-1-i, nrows-1-j)``; D is the number of consecutive non-empty rings from ring 0 - the first
  empty ring stops it, so a separate centre power / ground island does not count. A full array has ``D = ceil(min(ncols, nrows) / 2)``.
  Every numbered pad counts (the rule's, not the PCB agent's P6 escape estimate, which counts only pads on a net). The pads must be
  an array on the P grid (:func:`grid_misfit`: the smallest pad spacing one P, every centre on a grid point); otherwise D is not read
  (a note says why) and no row that needs it matches.
* **Part rows** match a case-insensitive prefix with spaces removed on the part's MPN when the IR has one, else on its value, else on
  its symbol name (the design file's ``value`` / ``symbol``); the reason names the field that matched. A part on a capacitor /
  resistor / inductor footprint (:data:`PASSIVE_LIBRARIES`) matches no part row, and a row whose prefix other makers' part numbers
  share (``vendors``: TI's ``CC12``, also Yageo's MLCCs) needs the manufacturer field, when it is set, to name the row's maker.
  Families named only by series or package code (Renesas RA / RX, the ESP32 series, PolarFire FC484) need a part catalog that names
  them; this version has none, so those rows never match (``unresolved``, :data:`CATALOG_UNRESOLVED`) - a stated limit, never a
  guessed prefix.

Combination (:func:`part_layers`, :func:`board_layers`):

1. Per part: every matching row; the part start is the largest start of its specific rows (a part, package or board row beats a generic
   footprint row), else of its generic rows; the part floor is the largest floor of all; then ``start = max(start, floor)``. A part no
   row matches contributes start 2 and floor 0 - deliberate: no opened source gives a package-level count for QFP / QFN / SOIC / SOT /
   TO / LGA families (the synthesis' gaps).
2. Per board: start = the largest part start, floor = the largest part floor and every extra floor the caller hands in (the PCB agent's
   P6 minimums: a part's sourced ``Component.min_pcb_layers`` and the area-array escape estimate); ``start = max(start, floor)``
   (Altera AN 613 sec. 1.4.1: the largest BGA sets the board minimum - the synthesis' citation, not re-opened here).
3. Both round up to the ladder :data:`~ai_eda.design.stackup.LAYER_ESCALATION` (NXP AN10778 sec. 5.2; ICD AN2015_9 p. 3 "The layer
   count always increments by even numbers"). A start above 12 is capped at 12 with a note; a floor above 12 refuses the plan.
4. Conditional rows (LPC55 with Ethernet, an Infineon QFN with P_D > 5 W, CC1101 + CC1190 on TI's reference design) apply only when
   the requirements state the condition in a form this version reads: ``power_dissipation`` (the confirmed requirement,
   :func:`~ai_eda.design.materials.read_power_dissipation`) is read; no requirement key names interfaces or a reference design, so the
   other two are skipped with that reason in the notes - never guessed from the netlist.

A board whose parts match no row with a start above 2 or a floor (``BoardLayers.applies`` false) is not touched by any of this: the PCB
agent routes it exactly as before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from ai_eda.compilers.schematic_layout import natural_ref_key
from ai_eda.design.materials import read_power_dissipation
from ai_eda.design.stackup import LAYER_ESCALATION
from ai_eda.ir import CircuitIR
from ai_eda.tools.kicad.library import KicadLibrary

PACKAGE_LAYERS_TOOL = "design.package_layers"
PACKAGE_LAYERS_VERSION = "0.1"
#: the project owner's rule as the notes and the kept stack's provenance quote it
PACKAGE_RULE = (
    "2026-11-13: start from the layer count commonly used for the board's IC packages (researched, sourced) and apply it first, then add "
    "layers when nets stay unrouted or remove layers while every net still routes, never below a manufacturer's floor"
)
#: where the table comes from
RESEARCH = "the multilayer plan's package-layer research (package_layers.json, 2026-11-13), every row's source opened there"
#: a footprint in this KiCad library is a BGA
BGA_LIBRARY = "Package_BGA"
#: a footprint whose name holds one of these as a token (split at ``-`` / ``_`` / spaces) is a BGA
BGA_TOKENS: frozenset[str] = frozenset({"BGA", "FBGA", "LFBGA", "TFBGA", "VFBGA", "UFBGA"})
#: a part on a footprint of a KiCad library whose name starts with one of these is a capacitor / resistor / inductor: no part row
#: (an IC's MPN prefix) matches it
PASSIVE_LIBRARIES: tuple[str, ...] = ("Capacitor_", "Resistor_", "Inductor_")
#: two lengths (mm) this close are one: KiCad's 1 nm resolution
LENGTH_TOLERANCE_MM = 1e-6
#: the ball depth is counted on the P grid only when the smallest pad spacing is within this fraction of P and every pad centre is
#: within it (in pitches) of a grid point: far above KiCad's 0.01 mm coordinate rounding (Package_BGA:BGA-352_..._P1.27mm steps
#: 1.26 to 1.28 mm), far below a half-pitch stagger or a ball spacing of twice P
GRID_TOLERANCE = 0.25
#: Infineon AN72845 sec. 4.6.1: "For systems with total power dissipation (PD) greater than 5 watts, use four layers."
PD_THRESHOLD_W = 5.0

#: the conditions a row may need, as the requirements state them
ETHERNET = "ethernet"
PD_OVER_5W = "pd_over_5w"
CC1190_REFERENCE = "cc1190_reference"
#: why the two conditions no requirement key of this version names cannot be read
UNREADABLE: dict[str, str] = {
    ETHERNET: ("no requirement key of this version names the board's interfaces, so 'the requirements name an Ethernet interface' cannot "
               "be read (it is never guessed from the netlist)"),
    CC1190_REFERENCE: ("no requirement key of this version names a reference design, so 'the design follows TI's CC1101-CC1190EM869RD "
                       "reference design' cannot be read (it is never guessed from the netlist)"),
}
#: why the rows that name a family only by series or package code never match in this version
CATALOG_UNRESOLVED = (
    "the source names the family by series or package code, not by an orderable MPN prefix, and this version has no part catalog that "
    "maps MPNs to it"
)

_PITCH_TOKEN = re.compile(r"^P(\d+(?:\.\d+)?)mm$")
_BODY_TOKEN = re.compile(r"^(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)mm$")


@dataclass(frozen=True)
class LayerRow:
    """One row of the table: what it matches, its start and floor, and its source, verbatim from the research (module docstring)."""

    id: str
    #: ``footprint`` (a generic package convention), ``package`` (an exact N / P / BODY), ``part`` (the part's own document), ``board``
    #: (several parts and a requirement together)
    kind: str
    start: int
    floor: int
    opened: bool
    #: ``False``: the quote came through an HTML extractor and its wording was not re-checked against the PDF
    quote_checked: bool
    match: str
    source: str
    quote: str
    conditions: str
    #: MPN / value / symbol prefixes (case-insensitive, spaces removed); a ``board`` row needs a part for each
    prefixes: tuple[str, ...] = ()
    #: the part's manufacturer field must contain one of these (case-insensitive)
    manufacturers: tuple[str, ...] = ()
    #: a prefix row whose prefix other makers' part numbers share: when the part's manufacturer field is set, it must name one of these
    #: (whole words, case-insensitive); the row's source is that maker's own document
    vendors: tuple[str, ...] = ()
    #: the footprint's KiCad library
    library: str | None = None
    bga: bool = False
    pitch_min: float | None = None
    #: ``True``: ``P > pitch_min`` (else ``P >= pitch_min``)
    pitch_min_open: bool = False
    pitch_max: float | None = None
    #: an exact pitch / body / numbered-pad count
    pitch: float | None = None
    body: tuple[float, float] | None = None
    pads: int | None = None
    depth_min: int | None = None
    depth_max: int | None = None
    #: a requirement condition (:data:`ETHERNET`, :data:`PD_OVER_5W`, :data:`CC1190_REFERENCE`)
    condition: str | None = None
    #: why this version cannot match the row (``None``: it can)
    unresolved: str | None = None

    @property
    def specific(self) -> bool:
        """A part, package or board row (it beats a generic footprint row for the part's start)."""
        return self.kind != "footprint"

    @property
    def footprint_bound(self) -> bool:
        return self.bga or self.library is not None or any(
            v is not None for v in (self.pitch_min, self.pitch_max, self.pitch, self.body, self.pads, self.depth_min, self.depth_max))

    def cite(self, *, quote: bool = False) -> str:
        """The row's source, and its quote (marked when not re-checked against the PDF)."""
        if not quote:
            return self.source
        checked = "" if self.quote_checked else " (quoted through an HTML extractor, not re-checked against the PDF)"
        return f'{self.source}: "{self.quote}"{checked}'


ROWS: tuple[LayerRow, ...] = (
    LayerRow(
        id="nxp_an10778_bga_d4", kind="footprint", start=2, floor=0, opened=True, quote_checked=True,
        bga=True, pitch_min=0.63, pitch_max=1.02, depth_max=4,
        match=(
            "BGA footprint (KiCad library Package_BGA, or footprint name token BGA/FBGA/LFBGA/TFBGA/VFBGA/UFBGA), 0.63 mm <= P <= "
            "1.02 mm, ball depth D <= 4"
        ),
        source=(
            "NXP AN10778 'PCB layout guidelines for NXP MCUs in BGA packages', Rev. 2, 15 April 2011, "
            "https://www.nxp.com/docs/en/application-note/AN10778.pdf, sec. 5.2 'How many PCB layers to fan-out the BGA' (p.11)"
        ),
        quote="Each additional BGA row will take one additional PCB layer to fan-out.",
        conditions=(
            "NXP rule: rows 1-2 escape on the layer the BGA sits on, rows 3-4 on the next signal layer, and each further row takes "
            "one more layer. The count includes power and ground. Round up to even: L = 1 if D <= 2, 2 if D <= 4, else D - 2. It "
            "holds for NXP's through-via dog-bone patterns at 1.0, 0.8 and 0.65 mm with one trace between vias (Fig 4/5). This is a "
            "start only. Text checked in the pdftotext extract."
        ),
    ),
    LayerRow(
        id="nxp_an10778_bga_d6", kind="footprint", start=4, floor=0, opened=True, quote_checked=True,
        bga=True, pitch_min=0.63, pitch_max=1.02, depth_min=5, depth_max=6,
        match="BGA footprint, 0.63 mm <= P <= 1.02 mm, D = 5 or 6",
        source="NXP AN10778 Rev. 2, 15 April 2011, https://www.nxp.com/docs/en/application-note/AN10778.pdf, sec. 5.2 (p.11)",
        quote="Each additional BGA row will take one additional PCB layer to fan-out.",
        conditions=(
            "L = D - 2, which is 3 or 4, rounded up to 4. The same through-via, one-trace-between-vias conditions apply. JLCPCB's BGA "
            "guide agrees ('A 0.8 mm BGA can be considered for a 4-layer or 6-layer stackup'). Start only."
        ),
    ),
    LayerRow(
        id="nxp_an10778_bga_d8", kind="footprint", start=6, floor=0, opened=True, quote_checked=True,
        bga=True, pitch_min=0.63, pitch_max=1.02, depth_min=7, depth_max=8,
        match="BGA footprint, 0.63 mm <= P <= 1.02 mm, D = 7 or 8",
        source=(
            "NXP AN10778 Rev. 2, 15 April 2011, https://www.nxp.com/docs/en/application-note/AN10778.pdf, sec. 5.2 (p.11), worked "
            "example TFBGA296"
        ),
        quote="the TFBGA296 has balls seven rows deep and will take 5 PCB layers to fan-out, including power and ground.",
        conditions="L = D - 2, which is 5 or 6, rounded up to 6. Start only.",
    ),
    LayerRow(
        id="nxp_an10778_bga_d10", kind="footprint", start=8, floor=0, opened=True, quote_checked=True,
        bga=True, pitch_min=0.63, pitch_max=1.02, depth_min=9, depth_max=10,
        match="BGA footprint, 0.63 mm <= P <= 1.02 mm, D = 9 or 10",
        source="NXP AN10778 Rev. 2, 15 April 2011, https://www.nxp.com/docs/en/application-note/AN10778.pdf, sec. 5.2 (p.11)",
        quote="Each additional BGA row will take one additional PCB layer to fan-out.",
        conditions="L = D - 2, which is 7 or 8, rounded up to 8. Start only.",
    ),
    LayerRow(
        id="nxp_an10778_bga_d12", kind="footprint", start=10, floor=0, opened=True, quote_checked=True,
        bga=True, pitch_min=0.63, pitch_max=1.02, depth_min=11, depth_max=12,
        match="BGA footprint, 0.63 mm <= P <= 1.02 mm, D = 11 or 12",
        source="NXP AN10778 Rev. 2, 15 April 2011, https://www.nxp.com/docs/en/application-note/AN10778.pdf, sec. 5.2 (p.11)",
        quote="Each additional BGA row will take one additional PCB layer to fan-out.",
        conditions="L = D - 2, which is 9 or 10, rounded up to 10. Start only.",
    ),
    LayerRow(
        id="nxp_an10778_bga_d13", kind="footprint", start=12, floor=0, opened=True, quote_checked=True,
        bga=True, pitch_min=0.63, pitch_max=1.02, depth_min=13,
        match="BGA footprint, 0.63 mm <= P <= 1.02 mm, D >= 13",
        source="NXP AN10778 Rev. 2, 15 April 2011, https://www.nxp.com/docs/en/application-note/AN10778.pdf, sec. 5.2 (p.11)",
        quote="Each additional BGA row will take one additional PCB layer to fan-out.",
        conditions=(
            "D = 13 or 14 gives 11 or 12, so the start is 12. D >= 15 gives 13 or more, which is above the ladder: the start is "
            "capped at 12 and the report says the convention exceeds the ladder. Start only."
        ),
    ),
    LayerRow(
        id="ns_an1126_bga_p127", kind="footprint", start=4, floor=0, opened=True, quote_checked=True,
        bga=True, pitch_min=1.02, pitch_min_open=True, pitch_max=1.30, depth_min=3,
        match="BGA footprint, 1.02 mm < P <= 1.30 mm (1.27 mm pitch), D >= 3",
        source=(
            "National Semiconductor AN-1126 'BGA (Ball Grid Array)', September 1999, mirror "
            "https://bitsavers.mirrorservice.org/components/national/_appNotes/AN-1126.pdf, 'Routing Guidelines' and Table 5 (p.2); "
            "supported by Motorola AN1231/D 'Plastic Ball Grid Array (PBGA)', "
            "https://www.nxp.com/docs/en/application-note/AN1231.pdf, escape routing section (p.3-4)"
        ),
        quote="Either a 1.0 or 1.27 mm pitch PBGA with four rows of solder balls can be routed to a four layer PCB",
        conditions=(
            "AN-1126 assumes a 4-layer board with 2 signal and 2 ground layers and NSMD pads. Four rows route at 0.15 mm line/space. "
            "Five rows at 1.27 mm need 0.125 mm L/S, with rows 1-3 on one signal layer and rows 4-5 on the second. Six rows need "
            "0.075 mm L/S and micro-vias. Motorola AN1231 also routes 4 outer rows at 1.27 mm on two signal layers, with power and "
            "ground to planes. This row is limited to P > 1.0 mm; the 1.0 mm case is left to the NXP rows (see the conflicts in "
            "rules). Start only. Text checked in the extracts."
        ),
    ),
    LayerRow(
        id="jlcpcb_bga_p05", kind="footprint", start=6, floor=0, opened=True, quote_checked=True,
        bga=True, pitch_min=0.48, pitch_max=0.52, depth_min=3,
        match="BGA footprint, 0.48 mm <= P <= 0.52 mm, D >= 3",
        source=(
            "JLCPCB blog 'BGA PCB Design Complete Guide: Layout and Routing Guidelines' (Jul 15, 2026, updated Nov 09, 2026), "
            "https://jlcpcb.com/blog/bga-pcb-design-complete-guide-layout-and-routing-guidelines, 'Stackup Selection and Layer Count "
            "for BGA PCB' (re-fetched in this step); the D >= 3 boundary is from NXP AN10778 Rev. 2, sec. 4.3 (p.10)"
        ),
        quote="a 0.5 mm BGA is typically better suited for a 6-layer, 8-layer or HDI plan",
        conditions=(
            "The start is the lowest of JLCPCB's '6-layer, 8-layer or HDI'. D >= 3 because NXP AN10778 sec. 4.3 says that at 0.5 mm "
            "with 0.08 mm trace/space the two outer rows escape without a fan-out via, and only the inner rows need vias and other "
            "layers. A footprint with D <= 2 matches no row. The same JLCPCB page says '4+ layer routing ... may not be necessary' "
            "with via-in-pad, so the removal step may reach 4 or 2. This is a fab blog, so it is a start only. Part rows override it "
            "(LPC55, MSP432E4, Renesas RA/RX, AM62x)."
        ),
    ),
    LayerRow(
        id="st_an4488_stm32f4", kind="part", start=4, floor=4, opened=True, quote_checked=True,
        prefixes=("STM32F4",),
        match="Part: MPN prefix 'STM32F4' (STM32F4xxxx family, any package)",
        source=(
            "STMicroelectronics AN4488 'Getting started with STM32F4xxxx MCU hardware development', Rev 7 (03-Oct-2018), sec. 8.1 "
            "'PCB stack-up', p.36/50, "
            "https://www.st.com/resource/en/application_note/an4488-getting-started-with-stm32f4xxxx-mcu-hardware-development-stmicroelectronics.pdf "
            "(opened in the in-app browser; WebFetch to st.com failed)"
        ),
        quote="The minimum configuration that can be used is 4 or 6 layers stack-up.",
        conditions=(
            "ST calls this the minimum, so it is a floor. Section 8.1 introduces it with impedance matching of high-speed signals; ST "
            "does not exclude simple boards, so the floor is applied to every STM32F4 board. ST also says 'An 8 layers boards may be "
            "required' for very dense boards with several external memories or an LCD. That is hedged and is not applied."
        ),
    ),
    LayerRow(
        id="st_an2586_stm32f10", kind="part", start=4, floor=0, opened=True, quote_checked=True,
        prefixes=("STM32F10",),
        match="Part: MPN prefix 'STM32F10' (STM32F100/F101/F102/F103/F105/F107, any package)",
        source=(
            "STMicroelectronics AN2586 'Getting started with STM32F10xxx hardware development', Rev 8 (09-Dec-2022), sec. 6.1 "
            "'Printed circuit board', p.21/29, "
            "https://www.st.com/resource/en/application_note/an2586-getting-started-with-stm32f10xxx-hardware-development-stmicroelectronics.pdf "
            "(opened in the in-app browser)"
        ),
        quote="it is best to use a multilayer printed circuit board (PCB) with a separate layer",
        conditions=(
            "The wording is hedged ('best'), and the same paragraph says economics often rule it out ('For many applications, "
            "economical reasons prohibit the use of this type of board'). So this is a start only. The value 4 is inferred: a "
            "dedicated VSS layer plus a dedicated VDD layer plus signals is at least 3 layers, so 4 on the even ladder."
        ),
    ),
    LayerRow(
        id="mchp_an2519_avr", kind="part", start=4, floor=0, opened=True, quote_checked=True,
        prefixes=("ATmega", "ATxmega"),
        match="Part: MPN prefix 'ATmega' or 'ATxmega' (Microchip/Atmel AVR MCUs, any package)",
        source=(
            "Microchip AN2519 'AVR Microcontroller Hardware Design Considerations', DS00002519B (Rev B, 02/2018), sec. 7 'Example "
            "Layout of ATxmega32A4 and ATmega324PB Devices', p.19, "
            "https://ww1.microchip.com/downloads/en/Appnotes/AN2519-AVR-Microcontroller-Hardware-Design-Considerations-00002519B.pdf"
        ),
        quote="recommended to have a multilayer design with supply and ground plane on separate layers",
        conditions=(
            "'Multilayer' is given no number, so this is a start only and 4 is inferred. The note gives no explicit 2-layer "
            "allowance. This is a candidate the owner may promote to floor 4. The ATmega128 is not named in the note (its examples "
            "are the ATxmega32A4 and ATmega324PB). ATtiny was not checked."
        ),
    ),
    # the quote came through an HTML extractor (WebFetch / a web page), kept verbatim as given; its wording was not re-checked against the PDF
    LayerRow(
        id="espressif_esp32", kind="part", start=4, floor=4, opened=True, quote_checked=False,
        library="Package_DFN_QFN", unresolved=CATALOG_UNRESOLVED,
        match=(
            "Part: Espressif ESP32-series bare chip (the series covered by the ESP32 edition of the guide; ESP32-S/C/H/P series have "
            "their own editions, which were not opened) in a Package_DFN_QFN footprint; not an RF_Module footprint"
        ),
        source=(
            "Espressif 'ESP Hardware Design Guidelines' for ESP32, latest web edition (footer (c) 2023-2026, commit 192b0a3), PCB "
            "Layout Design > General Principles of PCB Layout for the Chip, "
            "https://docs.espressif.com/projects/esp-hardware-design-guidelines/en/latest/esp32/pcb-layout-design.html"
        ),
        quote="It is recommended to use a four-layer PCB design:",
        conditions=(
            "Layer use: L1 signals and components, L2 solid GND, L3 power plus a few signals, L4 a few signals. Under the owner's "
            "rule this explicit recommendation is a floor. The same page also says 'A two-layer PCB design can also be used' (full "
            "GND under the chip, RF and crystal; RF impedance recomputed). That 2-layer option is used only if the user asks for "
            "pcb_layers=2; the router never removes layers below 4. The part catalog must name the series, because no MPN pattern "
            "came from a source."
        ),
    ),
    LayerRow(
        id="nxp_mimxrt105060hdug", kind="part", start=4, floor=4, opened=True, quote_checked=True,
        prefixes=("MIMXRT105", "MIMXRT106"), bga=True,
        match="Part: MPN prefix 'MIMXRT105' or 'MIMXRT106' (NXP i.MX RT1050 / RT1060; RT1064 per the guide's introduction), BGA",
        source=(
            "NXP 'Hardware Development Guide for the MIMXRT1050/MIMXRT1060 Processor', MIMXRT105060HDUG Rev.4 (10/2019), sec. 7.1 "
            "'Stackup', p.17, NXP community copy "
            "https://community.nxp.com/pwmxy87654/attachments/pwmxy87654/imxrt/23845/1/MIMXRT105060HDUG.pdf"
        ),
        quote="The recommended stackup is four layers",
        conditions=(
            "High-speed signals need reference planes on adjacent layers. The EVKB stack-up routes SDRAM on the top and bottom layers "
            "over GND. A newer revision (listed as January 2025) is behind a login and was not opened."
        ),
    ),
    LayerRow(
        id="ti_sprs717_am335x", kind="part", start=4, floor=4, opened=True, quote_checked=True,
        prefixes=("AM335",),
        match="Part: MPN prefix 'AM335' (TI Sitara AM3351-AM3359, any package)",
        source=(
            "TI SPRS717L 'AM335x Sitara Processors' data sheet (Oct 2011, rev. Mar 2020), sec. 7.7.2.1.2.3 'PCB Stackup' p.150 "
            "(LPDDR), 7.7.2.2.2.3 p.161 (DDR2), 7.7.2.3.3.3 p.174 (DDR3), https://www.ti.com/lit/ds/symlink/am3358.pdf"
        ),
        quote="The minimum stackup required for routing the AM335x device is a 4-layer stackup",
        conditions=(
            "The minimum is 2 signal routing layers, a full GND plane and a split power plane, with no ground-plane cuts in the DDR "
            "routing region. TI says more layers may be needed for other circuitry, SI/EMI or a smaller board. The ZCE Via Channel "
            "package can route on 4 layers total, but TI warns 4 layers may not meet performance goals (sec. 9.1, p.244)."
        ),
    ),
    # the quote came through an HTML extractor (WebFetch / a web page), kept verbatim as given; its wording was not re-checked against the PDF
    LayerRow(
        id="ti_spruiy5a_am64x_am243x", kind="part", start=6, floor=6, opened=True, quote_checked=False,
        prefixes=("AM64", "AM243"), bga=True, pitch=0.8, body=(17.2, 17.2),
        match=(
            "Part: MPN prefix 'AM64' or 'AM243' (TI AM64x / AM243x) AND BGA footprint with BODY 17.2x17.2 mm, P = 0.8 mm (21x21 ball "
            "array)"
        ),
        source=(
            "TI SPRUIY5A AM64x/AM243x BGA escape routing (Feb 2021, rev. Jul 2021), sec. 1 'Stackup', "
            "https://ti.com/document-viewer/lit/html/SPRUIY5A/GUID-B35E9170-8EF7-47A5-A1F5-156A7D072C39"
        ),
        quote="designs must have two routing layers, not counting the top and bottom layers",
        conditions=(
            "Through-vias only. TI requires 4 signal-capable layers (top, bottom and 2 inner). The smallest project stack with 4 "
            "signal layers is 6 (F.Cu, In2.Cu, In3.Cu and B.Cu signal; In1.Cu ground; In4.Cu power), so the floor is 6. The TI EVM's "
            "10-layer stack is an example, not a start. The quote came through the WebFetch extractor and should be re-checked "
            "verbatim."
        ),
    ),
    # the quote came through an HTML extractor (WebFetch / a web page), kept verbatim as given; its wording was not re-checked against the PDF
    LayerRow(
        id="ti_sprad13a_am62x", kind="part", start=6, floor=6, opened=True, quote_checked=False,
        prefixes=("AM623", "AM625"), bga=True, pitch=0.5, body=(13.0, 13.0),
        match="Part: MPN prefix 'AM623' or 'AM625' (TI AM62x) AND BGA footprint with BODY 13x13 mm, P = 0.5 mm (Via Channel Array)",
        source=(
            "TI SPRAD13A (May 2022 - Dec 2024), sec. 3 'Width/Spacing Proposal for Escapes', Table 3-1, "
            "https://ti.com/document-viewer/lit/html/SPRAD13/GUID-3BACCCE0-07CF-43A1-9F4F-99655150097B"
        ),
        quote="PCB layers (signal routing, total) recommended | 2, 6",
        conditions=(
            "TI recommends 2 signal routing layers and 6 layers total, with a minimum via diameter of 18 mil and a 10 mil hole. The "
            "table row came through the WebFetch extractor and should be re-checked."
        ),
    ),
    LayerRow(
        id="ti_slaa770_msp432e4", kind="part", start=4, floor=4, opened=True, quote_checked=True,
        prefixes=("MSP432E4",),
        match="Part: MPN prefix 'MSP432E4' (TI SimpleLink MSP432E4, any package incl. 212-ball BGA ZAD and 128-pin TQFP PDT)",
        source=(
            "TI SLAA770 'System Design Guidelines for SimpleLink MSP432E4 Microcontrollers', October 2017, "
            "https://www.ti.com/lit/pdf/slaa770, sec. 3.2.1 'Four-Layer Stack-up' (p.6) and sec. 3.3.3 '212-Ball BGA Escape Routing' "
            "(p.12)"
        ),
        quote="A four-layer stack-up (two signal layers, two power planes) is recommended for most designs.",
        conditions=(
            "For the BGA, sec. 3.3.3 says it 'can be routed using a minimum of a four-layer board' (4 mil trace, 4 mil space, 18 mil "
            "vias). Sec. 3.2.2 says 'A two-layer stack-up may be acceptable' only with no timing-sensitive high-speed interfaces, "
            "USB/Ethernet unused or on short runs, and adequate power, decoupling and ESD. That 2-layer option is used only if the "
            "user asks for pcb_layers=2. Text checked in the extract."
        ),
    ),
    LayerRow(
        id="ti_slaa530b_msp430", kind="part", start=4, floor=0, opened=True, quote_checked=True,
        prefixes=("MSP430",),
        match="Part: MPN prefix 'MSP430' (TI MSP430 MCUs, any package)",
        source=(
            "TI SLAA530B 'MSP430 System-Level ESD Considerations' (March 2012, rev. July 2021), sec. 2.2 'PCB Design and Layout', "
            "p.11 and p.13, https://www.ti.com/lit/pdf/slaa530"
        ),
        quote="If possible, use multi-layer PCBs with paired power and ground planes.",
        conditions=(
            "The wording is hedged ('If possible'), and the same section gives a 2-layer method (one layer mostly solid ground, or "
            "star grounding). So this is a start only. 4 is inferred from 'paired power and ground planes' plus signal layers."
        ),
    ),
    LayerRow(
        id="nxp_an13033_lpc55", kind="part", start=4, floor=0, opened=True, quote_checked=True,
        prefixes=("LPC55",),
        match="Part: MPN prefix 'LPC55' (NXP LPC55(S)xx, any package: HVQFN48/64, HTQFP64, HLQFP100, VFBGA98, VBGA196)",
        source=(
            "NXP AN13033 'Hardware Design Guidelines for LPC55(S)xx Microcontrollers', Rev. 2.2, 11 June 2025, "
            "https://www.nxp.com/docs/en/application-note/AN13033.pdf, sec. 9.3 'Printed circuit board'; NXP AN12581 '2-layer PCB "
            "Layer Design Guide for VFBGA98 Package', Rev. 0, 10/2019, https://www.nxp.com/docs/en/application-note/AN12581.pdf, sec. "
            "1 (p.1)"
        ),
        quote="For technical reasons, it is best to use a multilayer Printed Circuit Board (PCB) with a separate layer dedicated",
        conditions=(
            "The wording is hedged ('best'), and the same paragraph says economic reasons often prohibit it. AN12581 says VFBGA98 "
            "boards are 'Usually' 4-layer but shows a 2-layer route at 3.2 mil L/S. So this is a start only, and it overrides the "
            "generic 0.5 mm BGA start of 6. Text checked in the extracts."
        ),
    ),
    LayerRow(
        id="nxp_an13033_lpc55_ethernet", kind="part", start=4, floor=4, opened=True, quote_checked=True,
        prefixes=("LPC55",), condition=ETHERNET,
        match="Part: MPN prefix 'LPC55' AND the requirements name an Ethernet interface",
        source=(
            "NXP AN13033 Rev. 2.2, 11 June 2025, https://www.nxp.com/docs/en/application-note/AN13033.pdf, sec. 9.4.4 'PCB layer "
            "stacking'"
        ),
        quote="a four-layer PCB is recommended for implementing Ethernet applications and systems.",
        conditions=(
            "Conditional floor. It applies only when the requirements state Ethernet; it is never guessed from the netlist. Text "
            "checked in the extract."
        ),
    ),
    LayerRow(
        id="renesas_r01an7402_ra_rx_bga", kind="part", start=4, floor=4, opened=True, quote_checked=True,
        bga=True, unresolved=CATALOG_UNRESOLVED,
        match=(
            "Part: Renesas RA-family or RX-family MCU (series name as the guide's package table lists it, e.g. RA6M5, RX65N) in a BGA "
            "footprint (Package_BGA: BGA / LFBGA / TFBGA)"
        ),
        source=(
            "Renesas R01AN7402EJ0200 'RA family, RX family Board design guideline for BGA products', Rev.2.00, Sep.17.25, "
            "https://renesas.com/en/document/apn/board-design-guideline-bga-products, sec. 3.3 'Recommended layer configuration' "
            "(p.8) and sec. 3.4"
        ),
        quote="As for the layer configuration of the board, a 4-layer board is recommended",
        conditions=(
            "Layer use: L1 MCU, signals and VSS; L2 VSS plane; L3 power plane plus VSS; L4 signals. Two more layers may be added "
            "between L2 and L3 for better electrical performance. If no trace fits between balls, the board 'may be 6 or more' "
            "(context, not applied). The guide covers 0.5, 0.65 and 0.8 mm BGAs from 36 to 303 pins; this floor overrides the generic "
            "0.5 mm start of 6. The part catalog must map orderable MPNs to the RA/RX series, because no MPN prefix came from a "
            "source. Text checked in the extract."
        ),
    ),
    LayerRow(
        id="nxp_an10778_tfbga296", kind="package", start=6, floor=6, opened=True, quote_checked=True,
        bga=True, pads=296, pitch=0.8, body=(15.0, 15.0),
        match="Package: BGA footprint with N = 296, P = 0.8 mm, BODY 15x15 mm (NXP TFBGA296, SOT1048-1)",
        source="NXP AN10778 Rev. 2, 15 April 2011, https://www.nxp.com/docs/en/application-note/AN10778.pdf, sec. 5.2 (p.11)",
        quote=(
            "a PCB using the TFBGA296 package would require a minimum of six layers, including one split power plane and one ground "
            "plane."
        ),
        conditions=(
            "Through-vias, with one trace between pads. Six layers means 4 signal layers plus 2 planes, which matches the project's "
            "6-layer stack. Text checked in the extract."
        ),
    ),
    # the quote came through an HTML extractor (WebFetch / a web page), kept verbatim as given; its wording was not re-checked against the PDF
    LayerRow(
        id="mchp_an5672_polarfire_fc484", kind="part", start=10, floor=10, opened=True, quote_checked=False,
        unresolved=CATALOG_UNRESOLVED,
        match=(
            "Part: Microchip PolarFire FPGA whose package code (as the source names it) is 'FC484' (1.0 mm pitch, 484 balls); "
            "'FCV484' and 'FCG484' do not match"
        ),
        source=(
            "Microchip AN5672 'PolarFire FPGA Package Fanout Application Note', Rev. A (11/2024), sec. 5 'Stack-Up Design', Table "
            "5-1, "
            "https://onlinedocs.microchip.com/oxy/GUID-7FDF1450-875E-44EE-B2F5-78157F0C7F0D-en-US-1/GUID-E9515BAB-5F5A-42C5-8741-D1045AC6FBD5.html"
        ),
        quote="The layers required for FC484 package are the following: Signal layers: 4; Power plane: 2; Ground layer: 4",
        conditions=(
            "Microchip calls these layers 'required', so this is a floor: 4 signal + 2 power + 4 ground = 10. Other PolarFire "
            "packages have no layer count in the text and are not covered. The quote came through the WebFetch extractor."
        ),
    ),
    # the quote came through an HTML extractor (WebFetch / a web page), kept verbatim as given; its wording was not re-checked against the PDF
    LayerRow(
        id="ti_snvsbd7b_lmr36510", kind="part", start=4, floor=4, opened=True, quote_checked=False,
        prefixes=("LMR36510",),
        match="Part: MPN prefix 'LMR36510' (TI 1-A 65-V buck, HSOIC-8 DDA)",
        source=(
            "TI LMR36510 data sheet SNVSBD7B (Rev B, Nov 2025), sec. 8.5.1.1 'Ground and Thermal Considerations', "
            "https://www.ti.com/document-viewer/lit/html/SNVSBD7B/GUID-5352DB0B-7A31-4F59-B4FF-FA3BDFD862DD"
        ),
        quote="Use a four-layer board with the copper thickness for the four layers",
        conditions=(
            "Copper 2 oz / 1 oz / 1 oz / 2 oz. One middle layer is a solid ground plane. The project's generic stacks use 35 um outer "
            "and 17.5 um inner copper, which does not match TI's copper weights; the report must say so. The quote came through the "
            "WebFetch extractor."
        ),
    ),
    # the quote came through an HTML extractor (WebFetch / a web page), kept verbatim as given; its wording was not re-checked against the PDF
    LayerRow(
        id="ti_snvsar0_lm53602_lm53603", kind="part", start=4, floor=4, opened=True, quote_checked=False,
        prefixes=("LM53602", "LM53603"),
        match="Part: MPN prefix 'LM53602' or 'LM53603' (TI buck, HTSSOP-16 PWP)",
        source=(
            "TI LM53602/LM53603 data sheet SNVSAR0 (Nov 2016), sec. 11.1.1 'Ground and Thermal Plane Considerations', "
            "https://ti.com/document-viewer/LM53603/datasheet/layout"
        ),
        quote=(
            "TI recommends using a four-layer board with the copper thickness, starting from the top, as: 2 oz. / 1 oz. / 1 oz. / 2 "
            "oz."
        ),
        conditions=(
            "At least one ground plane goes in a middle layer. The EP ties to it through at least a 4x4 array of 10 mil thermal vias. "
            "The copper weights differ from the project's generic stacks (report it). The quote came through the WebFetch extractor."
        ),
    ),
    # the quote came through an HTML extractor (WebFetch / a web page), kept verbatim as given; its wording was not re-checked against the PDF
    LayerRow(
        id="ti_snvsbi9a_lmr33610", kind="part", start=4, floor=4, opened=True, quote_checked=False,
        prefixes=("LMR33610",),
        match="Part: MPN prefix 'LMR33610' (TI buck; SOIC with thermal pad per the document)",
        source=(
            "TI LMR33610 data sheet SNVSBI9A (Oct 2019, rev. Jun 2022), sec. 11.1 'Layout Guidelines', "
            "https://www.ti.com/document-viewer/LMR33610/datasheet/GUID-C0E7E4C1-BB02-4620-B7CE-140F79A47FCF"
        ),
        quote="Use at least one ground plane in one of the middle layers.",
        conditions=(
            "No number is printed, but the wording forces one: a middle layer needs at least 3 layers, so 4 on the even ladder. TI "
            "also says multiple copper layers are 'recommended' and top/bottom copper should be 2 oz. The quote came through the "
            "WebFetch extractor."
        ),
    ),
    # the quote came through an HTML extractor (WebFetch / a web page), kept verbatim as given; its wording was not re-checked against the PDF
    LayerRow(
        id="ti_snvaa93_lmr38020_lmr38025", kind="part", start=4, floor=0, opened=True, quote_checked=False,
        prefixes=("LMR38020", "LMR38025"),
        match="Part: MPN prefix 'LMR38020' or 'LMR38025' (TI automotive buck)",
        source=(
            "TI SNVAA93 'Reduce Conducted EMI in Automotive Buck Converter Applications' (Dec 2023), sec. 3.2 'Layout Consideration', "
            "https://ti.com/document-viewer/lit/html/SNVAA93/GUID-B4AD942C-CE4C-44AD-B59B-68B10ADED91B"
        ),
        quote="when possible, a four-layer PCB is recommended.",
        conditions=(
            "The wording is hedged ('when possible'), so this is a start only. Layer use: L1 converter, inductor and SW node; L2 "
            "continuous GND; L3 input, output and signal routing; L4 EMI filter. The quote came through the WebFetch extractor."
        ),
    ),
    # the quote came through an HTML extractor (WebFetch / a web page), kept verbatim as given; its wording was not re-checked against the PDF
    LayerRow(
        id="ti_cc1101_cc1190em869rd", kind="board", start=4, floor=4, opened=True, quote_checked=False,
        prefixes=("CC1101", "CC1190"), condition=CC1190_REFERENCE,
        match=(
            "Board: contains MPN prefix 'CC1101' AND MPN prefix 'CC1190' AND the requirements state the design follows TI's "
            "CC1101-CC1190EM869RD reference design (869 MHz)"
        ),
        source=(
            "TI tool page 'CC1101-CC1190EM869RD reference design', https://www.ti.com/tool/CC1101-CC1190EM869RD; TI CC1190 data sheet "
            "SWRS089A (Nov 2009, rev. Feb 2010), p.4 NOTE, https://www.ti.com/lit/ds/symlink/cc1190.pdf"
        ),
        quote="This is a 4-layer reference design with a discrete balun",
        conditions=(
            "The CC1190 data sheet says 'Following the CC1190EM reference layout is recommended. Changes will alter the performance.' "
            "(checked in the extract). The 4-layer statement comes from the tool page, which the fetch tool truncated; re-check it. "
            "Other CC1190 bands and reference designs were not opened, so this row does not cover them."
        ),
    ),
    LayerRow(
        id="ti_swra407_cc112x_cc117x_cc12xx", kind="part", start=4, floor=0, opened=True, quote_checked=True,
        # 'CC12' also starts Yageo MLCC part numbers: Digi-Key lists CC1206KRX7R9BB104 as a YAGEO 0.1 uF 50 V X7R 1206 (3216 Metric)
        # ceramic capacitor (https://www.digikey.ie/en/products/detail/yageo/CC1206KRX7R9BB104/303089, a distributor listing opened
        # 2026-11-13; the Yageo datasheet did not load, so the part-number format is not checked against it). So a part whose
        # manufacturer field is set must be TI's, and no prefix row matches a passive footprint (PASSIVE_LIBRARIES)
        prefixes=("CC112", "CC117", "CC12"), vendors=("Texas Instruments", "TI"),
        match="Part: MPN prefix 'CC112', 'CC117' or 'CC12' (TI CC112x / CC117x / CC12xx, e.g. CC1200 VQFN-32)",
        source=(
            "TI Design Note DN039, SWRA407 (2012-08-20), sec. 1 (p.1) and sec. 4 / Table 15 (p.19), "
            "https://www.ti.com/lit/pdf/swra407; TI CC1200 data sheet SWRS123D (rev. Oct 2014), sec. 7.2 p.26, "
            "https://www.ti.com/lit/ds/symlink/cc1200.pdf"
        ),
        quote="presented with the standard 4-layer ref design [7] and with a 2-layer reference design [8].",
        conditions=(
            "The standard reference design is 4-layer, so start at 4. A 2-layer reference design with the Murata LFD21868MMF1D386 IPC "
            "exists at 868/915 MHz, so there is no floor. The stack must match the reference design that is copied (see the RF rule). "
            "Text checked in the extract."
        ),
    ),
    LayerRow(
        id="infineon_an72845_qfn", kind="part", start=4, floor=4, opened=True, quote_checked=True,
        manufacturers=("Infineon", "Cypress"), library="Package_DFN_QFN", condition=PD_OVER_5W,
        match=(
            "Part: Infineon (incl. former Cypress) device in a Package_DFN_QFN footprint AND the requirements state the system's "
            "total power dissipation PD > 5 W"
        ),
        source=(
            "Infineon AN72845 'Design guidelines for Infineon quad flat no-lead (QFN) packaged devices', 001-72845 Rev. *C, "
            "2022-08-29, https://www.infineon.com/an72845, sec. 4.6.1 'Number of PCB layers' (p.12)"
        ),
        quote="Recommendation: For systems with total power dissipation (PD) greater than 5 watts, use four layers.",
        conditions=(
            "Conditional floor, thermal only. It applies only when the requirements state PD > 5 W; it is never estimated. Infineon "
            "says 2 to 4 layers can lower Tj by up to 20 C, and suggests 2 oz outer copper above 3 W. Text checked in the extract."
        ),
    ),
)


def _check_rows(rows: Sequence[LayerRow]) -> None:
    """Run at import: only opened rows, unique ids, a known kind, counts on the ladder (0 = no floor), the floor never above the start."""
    ids = [r.id for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"package_layers: duplicate row ids {sorted(i for i in ids if ids.count(i) > 1)}")
    for r in rows:
        if not r.opened:
            raise ValueError(f"package_layers: row {r.id}'s source was not opened; only opened rows are kept")
        if r.kind not in ("footprint", "package", "part", "board"):
            raise ValueError(f"package_layers: row {r.id} has kind {r.kind!r}")
        if r.start not in LAYER_ESCALATION or (r.floor and r.floor not in LAYER_ESCALATION) or r.floor > r.start:
            raise ValueError(f"package_layers: row {r.id} start {r.start} / floor {r.floor}: both must be counts of the ladder {LAYER_ESCALATION} "
                             "(a floor of 0 = none) and the floor no higher than the start")


_check_rows(ROWS)


# --------------------------------------------------------------------------- footprint facts


@dataclass(frozen=True)
class FootprintFacts:
    """What the rows read of a footprint (module docstring: P, BODY, N, BGA, D)."""

    lib_id: str
    library: str
    name: str
    bga: bool
    pitch: float | None
    #: ``"name"`` (the ``P<x>mm`` token), ``"pads"`` (the smallest pad-centre spacing along x or y) or ``""`` (none)
    pitch_from: str
    body: tuple[float, float] | None
    pads: int
    #: the ball depth (``None``: not a BGA, no pitch, or not read - ``depth_why``)
    depth: int | None
    #: why a BGA's ball depth is not read: its pads are not on the P grid (:func:`grid_misfit`), ``None`` otherwise
    depth_why: str | None = None

    def describe(self) -> str:
        bits = [self.lib_id]
        if self.bga:
            bits.append("BGA")
        if self.pitch is not None:
            bits.append(f"P {self.pitch:g} mm" + (" (smallest pad spacing)" if self.pitch_from == "pads" else ""))
        if self.body is not None:
            bits.append(f"body {self.body[0]:g} x {self.body[1]:g} mm")
        bits.append(f"N {self.pads}")
        if self.depth is not None:
            bits.append(f"D {self.depth}")
        return ", ".join(bits)


def name_tokens(name: str) -> tuple[float | None, tuple[float, float] | None, bool]:
    """``(P, BODY, BGA token)`` read from a KiCad footprint name (``LFBGA-144_10x10mm_Layout12x12_P0.8mm`` -> 0.8, (10, 10), True)."""
    pitch = body = None
    for token in name.split("_"):
        if pitch is None and (m := _PITCH_TOKEN.match(token)):
            pitch = float(m.group(1))
        elif body is None and (m := _BODY_TOKEN.match(token)):
            body = (float(m.group(1)), float(m.group(2)))
    bga = any(t.upper() in BGA_TOKENS for t in re.split(r"[-_ ]", name))
    return pitch, body, bga


def pad_spacing(centres: Sequence[tuple[float, float]]) -> float | None:
    """The smallest positive spacing between two pad centres on one row (same y) along x, or on one column (same x) along y; ``None``."""
    best: float | None = None
    for axis in (0, 1):
        lines: dict[float, set[float]] = {}
        for c in centres:
            lines.setdefault(round(c[1 - axis], 6), set()).add(round(c[axis], 6))
        for values in lines.values():
            ordered = sorted(values)
            for a, b in zip(ordered, ordered[1:]):
                d = round(b - a, 6)
                if d > 0 and (best is None or d < best):
                    best = d
    return best


def ball_depth(centres: Sequence[tuple[float, float]], pitch: float) -> int:
    """D: the number of consecutive non-empty rings from the array's edge (module docstring); 0 for no centres."""
    if not centres or pitch <= 0:
        return 0
    x0, y0 = min(x for x, _ in centres), min(y for _, y in centres)
    cells = {(round((x - x0) / pitch), round((y - y0) / pitch)) for x, y in centres}
    ncols, nrows = max(i for i, _ in cells) + 1, max(j for _, j in cells) + 1
    rings = {min(i, j, ncols - 1 - i, nrows - 1 - j) for i, j in cells}
    depth = 0
    while depth in rings:
        depth += 1
    return depth


def grid_misfit(centres: Sequence[tuple[float, float]], pitch: float) -> str | None:
    """Why ``centres`` are not an array on the ``pitch`` grid that :func:`ball_depth` counts on (``None``: they are). Two misfits
    (:data:`GRID_TOLERANCE`): the smallest pad spacing is not one pitch (KiCad's ``BGA-152_14x18mm_Layout13x17_P0.5mm`` has its pads
    1.0 mm apart: on the 0.5 mm grid every odd ring is empty and D would be 1), or pad centres lie off the grid (a staggered or
    two-pitch array, e.g. ``BGA-200_10x14.5mm_Layout12x22_P0.8x0.65mm``, whose two-axis pitch token is not read)."""
    if not centres or pitch <= 0:
        return None
    spacing = pad_spacing(centres)
    if spacing is not None and abs(spacing / pitch - 1) > GRID_TOLERANCE:
        return f"the smallest pad spacing is {spacing:g} mm, not the pitch of {pitch:g} mm"
    x0, y0 = min(x for x, _ in centres), min(y for _, y in centres)
    off = sum(1 for x, y in centres if any(abs(v / pitch - round(v / pitch)) > GRID_TOLERANCE for v in (x - x0, y - y0)))
    if off:
        return f"{off} of {len(centres)} pad centres lie more than {GRID_TOLERANCE:g} pitch off the {pitch:g} mm grid"
    return None


def footprint_facts(lib_id: str, name: str, pads: Sequence) -> FootprintFacts:
    """The facts of a footprint from its library id, name and pads (:class:`~ai_eda.tools.kicad.library.Pad`: ``number``, ``pad_type``,
    ``layers``, ``x``, ``y``). Only numbered copper pads count (no NPTH hole, no paste-only aperture); a number seen twice is one pad.
    A BGA whose pads are not on its P grid (:func:`grid_misfit`) has no ball depth (``depth_why`` says why): never a depth counted on
    the wrong grid."""
    library = lib_id.split(":", 1)[0] if ":" in lib_id else ""
    copper = [p for p in pads if p.number and p.pad_type != "np_thru_hole" and any(layer.endswith(".Cu") for layer in p.layers)]
    centres = sorted({(round(p.x, 6), round(p.y, 6)) for p in copper})
    pitch, body, token = name_tokens(name)
    pitch_from = "name" if pitch is not None else ""
    if pitch is None:
        pitch = pad_spacing(centres)
        pitch_from = "pads" if pitch is not None else ""
    bga = library == BGA_LIBRARY or token
    depth = depth_why = None
    if bga and pitch is not None:
        depth_why = grid_misfit(centres, pitch)
        depth = ball_depth(centres, pitch) if depth_why is None else None
    return FootprintFacts(lib_id=lib_id, library=library, name=name, bga=bga, pitch=pitch, pitch_from=pitch_from, body=body,
                          pads=len({p.number for p in copper}), depth=depth, depth_why=depth_why)


# --------------------------------------------------------------------------- matching


def _norm(text: str) -> str:
    return "".join(text.split()).casefold()


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= LENGTH_TOLERANCE_MM


def _names(manufacturer: str, vendor: str) -> bool:
    """``manufacturer`` names ``vendor`` as whole words, case-insensitive ('TI' in 'TI Inc', not in 'Lattice')."""
    return re.search(rf"(?<![0-9a-z]){re.escape(vendor.casefold())}(?![0-9a-z])", manufacturer.casefold()) is not None


def identities(mpn: str | None, value: str | None, symbol: str | None) -> list[tuple[str, str]]:
    """The fields a part row's prefix is matched on, in order: the MPN alone when there is one, else the value, then the symbol name."""
    if mpn is not None and mpn.strip():
        return [("MPN", mpn)]
    return [(f, t) for f, t in (("value", value), ("symbol", symbol)) if t is not None and t.strip()]


def _prefix(ids: Sequence[tuple[str, str]], prefixes: Sequence[str]) -> str | None:
    """``"value 'STM32F407VET6' starts with 'STM32F4'"`` for the first field and prefix that match, else ``None``."""
    for fld, text in ids:
        for p in prefixes:
            if _norm(text).startswith(_norm(p)):
                return f"{fld} {text!r} starts with {p!r}"
    return None


def _footprint_miss(row: LayerRow, fp: FootprintFacts | None) -> str | None:
    """What of ``row``'s footprint match ``fp`` misses (``None``: it matches)."""
    if not row.footprint_bound:
        return None
    if fp is None:
        return "no footprint verified in the KiCad library"
    if row.library is not None and fp.library != row.library:
        return f"not in library {row.library}"
    if row.bga and not fp.bga:
        return "not a BGA footprint"
    p = fp.pitch
    if row.pitch is not None and (p is None or not _same(p, row.pitch)):
        return f"pitch is not {row.pitch:g} mm"
    if row.pitch_min is not None or row.pitch_max is not None:
        if p is None:
            return "no pitch"
        if row.pitch_min is not None and (p <= row.pitch_min + LENGTH_TOLERANCE_MM if row.pitch_min_open else p < row.pitch_min - LENGTH_TOLERANCE_MM):
            return "pitch below the row's range"
        if row.pitch_max is not None and p > row.pitch_max + LENGTH_TOLERANCE_MM:
            return "pitch above the row's range"
    if row.body is not None:
        b = fp.body
        if b is None or not ((_same(b[0], row.body[0]) and _same(b[1], row.body[1])) or (_same(b[0], row.body[1]) and _same(b[1], row.body[0]))):
            return f"body is not {row.body[0]:g} x {row.body[1]:g} mm"
    if row.pads is not None and fp.pads != row.pads:
        return f"not {row.pads} numbered pads"
    if row.depth_min is not None or row.depth_max is not None:
        d = fp.depth
        if d is None:
            return "no ball depth"
        if (row.depth_min is not None and d < row.depth_min) or (row.depth_max is not None and d > row.depth_max):
            return "ball depth outside the row's range"
    return None


@dataclass(frozen=True)
class BoardConditions:
    """What the requirements state that a conditional row reads (:data:`PD_OVER_5W`; the other two are unreadable, :data:`UNREADABLE`)."""

    #: the confirmed ``power_dissipation`` in W and its requirement id; ``None`` with ``power_dissipation_why`` when unusable, with
    #: ``power_dissipation_why`` ``None`` when not stated
    power_dissipation_w: float | None = None
    power_dissipation_req: str = ""
    power_dissipation_why: str | None = None


def condition_holds(condition: str, conditions: BoardConditions) -> tuple[bool | None, str]:
    """``(True / False, what was read)`` or ``(None, why it cannot be read)`` for one row condition."""
    if condition == PD_OVER_5W:
        if conditions.power_dissipation_w is None:
            return None, (f"the power_dissipation requirement is unusable ({conditions.power_dissipation_why})" if conditions.power_dissipation_why
                          else "the requirements state no power_dissipation (the system's total P_D is never estimated)")
        w = conditions.power_dissipation_w
        return w > PD_THRESHOLD_W, f"{conditions.power_dissipation_req} states P_D = {w:g} W ({'>' if w > PD_THRESHOLD_W else '<='} {PD_THRESHOLD_W:g} W)"
    return None, UNREADABLE.get(condition, f"condition {condition!r} is not read in this version")


@dataclass(frozen=True)
class RowMatch:
    """A row that matched a part (or a board): why (which field, which footprint facts, which requirement), and its start / floor."""

    ref: str
    row: LayerRow
    why: str

    def line(self, *, quote: bool = False) -> str:
        return f"{self.ref} ({self.why}): {self.row.id} start {self.row.start}, floor {self.row.floor or 'none'} - {self.row.cite(quote=quote)}"


@dataclass
class PartLayers:
    """One part's start and floor (2 / 0 when no row matches) and the rows that set them."""

    ref: str
    start: int = LAYER_ESCALATION[0]
    floor: int = 0
    matches: list[RowMatch] = field(default_factory=list)
    start_rows: list[RowMatch] = field(default_factory=list)
    floor_rows: list[RowMatch] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def applies(self) -> bool:
        """A row with a start above 2 or a floor matched (otherwise the part changes nothing)."""
        return self.start > LAYER_ESCALATION[0] or self.floor > 0


def part_layers(ref: str, ids: Sequence[tuple[str, str]], manufacturer: str | None, fp: FootprintFacts | None,
                conditions: BoardConditions | None = None, rows: Sequence[LayerRow] = ()) -> PartLayers:
    """One part's start and floor (module docstring: combination 1). ``ids``: :func:`identities`; ``rows`` defaults to :data:`ROWS`."""
    conditions = conditions or BoardConditions()
    out = PartLayers(ref)
    if fp is not None and fp.depth_why is not None:
        out.notes.append(f"{ref}: footprint {fp.lib_id}: the ball depth D is not read - {fp.depth_why}; the BGA rows that need it do not match")
    passive = fp is not None and fp.library.startswith(PASSIVE_LIBRARIES)
    for row in rows or ROWS:
        if row.unresolved is not None or row.kind == "board":
            continue
        why: list[str] = []
        if row.prefixes:
            hit = _prefix(ids, row.prefixes)
            if hit is None or passive:  # an IC's MPN prefix never names a capacitor, resistor or inductor
                continue
            if row.vendors and manufacturer is not None and manufacturer.strip() and not any(_names(manufacturer, v) for v in row.vendors):
                continue  # another maker's part number that shares the prefix: not the row's part
            why.append(hit)
        if row.manufacturers:
            m = (manufacturer or "").casefold()
            name = next((n for n in row.manufacturers if n.casefold() in m), None)
            if name is None:
                continue
            why.append(f"manufacturer {manufacturer!r} names {name}")
        miss = _footprint_miss(row, fp)
        if miss is not None:
            # the part is the row's (its MPN / value / symbol names it), its footprint is not: said, never matched; a row named by the
            # manufacturer alone (Infineon's QFN row) says nothing of the maker's other packages
            if row.prefixes:
                out.notes.append(f"{ref} ({'; '.join(why)}): {row.id} not applied - footprint {fp.lib_id if fp else 'none'}: {miss}")
            continue
        if row.footprint_bound and fp is not None:
            why.append(f"footprint {fp.describe()}")
        if row.condition is not None:
            holds, read = condition_holds(row.condition, conditions)
            if holds is None:
                out.notes.append(f"{ref} ({'; '.join(why)}): {row.id} (start {row.start}, floor {row.floor or 'none'}, {row.source}) skipped - {read}")
                continue
            if not holds:
                continue
            why.append(read)
        out.matches.append(RowMatch(ref, row, "; ".join(why)))
        if row.id == "nxp_an10778_bga_d13" and fp is not None and fp.depth is not None and fp.depth - 2 > LAYER_ESCALATION[-1]:
            out.notes.append(f"{ref}: D = {fp.depth} gives {fp.depth - 2} layers by NXP AN10778 sec. 5.2, which exceeds the ladder; the start "
                             f"is capped at {LAYER_ESCALATION[-1]}")
    specific = [m for m in out.matches if m.row.specific]
    pool = specific or out.matches
    if pool:
        out.start = max(m.row.start for m in pool)
        out.start_rows = [m for m in pool if m.row.start == out.start]
    floors = [m for m in out.matches if m.row.floor]
    if floors:
        out.floor = max(m.row.floor for m in floors)
        out.floor_rows = [m for m in floors if m.row.floor == out.floor]
    if out.floor > out.start:
        out.start, out.start_rows = out.floor, list(out.floor_rows)
    return out


def board_row_notes(parts_ids: Sequence[tuple[str, Sequence[tuple[str, str]]]], conditions: BoardConditions | None = None,
                    rows: Sequence[LayerRow] = ()) -> tuple[list[RowMatch], list[str]]:
    """The ``board`` rows: ``(matches, notes)``. A row whose every prefix names a part of the board applies when its condition holds;
    an unreadable condition is a note (``parts_ids``: ``(ref, identities)`` per part, in ref order)."""
    conditions = conditions or BoardConditions()
    matches: list[RowMatch] = []
    notes: list[str] = []
    for row in rows or ROWS:
        if row.kind != "board" or row.unresolved is not None:
            continue
        found: list[str] = []
        for p in row.prefixes:
            hit = next((f"{ref} {why}" for ref, ids in parts_ids if (why := _prefix(ids, (p,))) is not None), None)
            if hit is None:
                break
            found.append(hit)
        else:
            holds, read = condition_holds(row.condition, conditions) if row.condition else (True, "")
            if holds is None:
                notes.append(f"board ({'; '.join(found)}): {row.id} (start {row.start}, floor {row.floor or 'none'}, {row.source}) skipped - {read}")
            elif holds:
                matches.append(RowMatch("board", row, "; ".join([*found, *([read] if read else [])])))
    return matches, notes


def ladder_count(n: int) -> int | None:
    """The first count of :data:`~ai_eda.design.stackup.LAYER_ESCALATION` at or above ``n`` (``None``: above the ladder)."""
    return next((rung for rung in LAYER_ESCALATION if rung >= n), None)


@dataclass
class BoardLayers:
    """The board's start and floor on the ladder, why (each row with its source), and what was skipped."""

    #: the ladder count the search starts at (2 when nothing applies)
    start: int = LAYER_ESCALATION[0]
    #: the ladder count removal never goes below (0: no floor)
    floor: int = 0
    #: some part (or board row) matched a row with a start above 2 or a floor: the package search applies
    applies: bool = False
    #: one entry per row / extra floor that set the start, and the floor (the floor's with the quote)
    start_why: list[str] = field(default_factory=list)
    floor_why: list[str] = field(default_factory=list)
    #: provenance ``derived_from`` entries: ``package_layers:<ref>=<row id>`` per applying row, then the extra floors'
    derived: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: why the plan cannot be used (a floor above the ladder), else ``None``
    refusal: str | None = None
    parts: list[PartLayers] = field(default_factory=list)


def board_layers(parts: Sequence[PartLayers], extra_floors: Sequence[tuple[int, str, str]] = (), board_rows: Sequence[RowMatch] = (),
                 notes: Sequence[str] = ()) -> BoardLayers:
    """The board's start and floor (module docstring: combination 2 and 3). ``extra_floors``: ``(layers, reason, derived)`` each - the
    PCB agent's P6 minimums; ``board_rows``: the matched ``board`` rows; ``notes``: carried in front of the parts' own notes. A plan
    that does not apply carries no note at all (the board's notes stay as they were; each part's notes stay in ``parts``)."""
    out = BoardLayers(parts=list(parts))
    board_part = PartLayers("board")
    if board_rows:
        board_part.matches = list(board_rows)
        board_part.start = max(m.row.start for m in board_rows)
        board_part.start_rows = [m for m in board_rows if m.row.start == board_part.start]
        board_part.floor = max(m.row.floor for m in board_rows)
        board_part.floor_rows = [m for m in board_rows if m.row.floor and m.row.floor == board_part.floor]
    every = [*parts, *([board_part] if board_rows else [])]
    out.applies = any(p.applies for p in every)
    if out.applies:
        out.notes = [*notes, *(n for p in parts for n in p.notes)]
    start =max((p.start for p in every), default=LAYER_ESCALATION[0])
    floor = max([p.floor for p in every] + [n for n, _, _ in extra_floors] + [0])
    out.start_why = [m.line() for p in every if p.start == start for m in p.start_rows]
    out.floor_why = ([m.line(quote=True) for p in every if p.floor and p.floor == floor for m in p.floor_rows]
                     + [reason for n, reason, _ in extra_floors if n == floor and floor])
    if floor > start:
        start, out.start_why = floor, list(out.floor_why)
    out.derived = [f"package_layers:{m.ref}={m.row.id}" for p in every for m in p.matches if m.row.start > LAYER_ESCALATION[0] or m.row.floor]
    out.derived += [d for _, _, d in extra_floors]
    if floor:
        rung = ladder_count(floor)
        if rung is None:
            out.refusal = f"the floor of {floor} copper layers is above the ladder's last count {LAYER_ESCALATION[-1]} ({'; '.join(out.floor_why)})"
            return out
        if rung != floor and out.applies:  # a plan that does not apply says nothing: the board's notes stay as they were
            out.notes.append(f"the floor of {floor} copper layers rounds up to {rung}, the next count of the ladder")
        out.floor = rung
    rung = ladder_count(start)
    if rung is None:
        if out.applies:
            out.notes.append(f"the start of {start} copper layers exceeds the ladder; it is capped at {LAYER_ESCALATION[-1]}")
        rung = LAYER_ESCALATION[-1]
    out.start = max(rung, out.floor)
    return out


def board_conditions(ir: CircuitIR) -> BoardConditions:
    """The conditions the requirements state, as this version reads them (only ``power_dissipation``; module docstring)."""
    stated, why = read_power_dissipation(ir)
    if stated is None:
        return BoardConditions(power_dissipation_why=why)
    req, traced = stated
    return BoardConditions(power_dissipation_w=float(traced.value), power_dissipation_req=req.id)


def plan_for_ir(ir: CircuitIR, library: KicadLibrary, extra_floors: Sequence[tuple[int, str, str]] = ()) -> BoardLayers:
    """The board's :class:`BoardLayers` from ``ir``'s parts in natural ref order: each part's identities (:func:`identities`: its MPN,
    else its value and symbol name), manufacturer and verified footprint (read from ``library``); ``extra_floors`` as :func:`board_layers`.
    A footprint that is not verified in the library gives no footprint facts (rows that need them do not match)."""
    conditions = board_conditions(ir)
    parts: list[PartLayers] = []
    parts_ids: list[tuple[str, list[tuple[str, str]]]] = []
    for comp in sorted(ir.components, key=lambda c: natural_ref_key(c.ref)):
        ids = identities(comp.mpn.value if comp.mpn is not None else None, comp.value, comp.symbol.name if comp.symbol is not None else None)
        fp = None
        if comp.footprint is not None and library.resolve_footprint(comp.footprint).verified:
            f = library.load_footprint(comp.footprint)
            fp = footprint_facts(f.lib_id, f.name, f.pads)
        manufacturer = comp.manufacturer.value if comp.manufacturer is not None else None
        parts.append(part_layers(comp.ref, ids, manufacturer, fp, conditions))
        parts_ids.append((comp.ref, ids))
    board_rows, notes = board_row_notes(parts_ids, conditions)
    return board_layers(parts, extra_floors, board_rows, notes)


__all__ = [
    "BGA_LIBRARY", "BGA_TOKENS", "BoardConditions", "BoardLayers", "CATALOG_UNRESOLVED", "CC1190_REFERENCE", "ETHERNET", "FootprintFacts",
    "GRID_TOLERANCE", "LENGTH_TOLERANCE_MM", "LayerRow", "PACKAGE_LAYERS_TOOL", "PACKAGE_LAYERS_VERSION", "PACKAGE_RULE", "PASSIVE_LIBRARIES",
    "PD_OVER_5W", "PD_THRESHOLD_W", "PartLayers", "RESEARCH", "ROWS", "RowMatch", "UNREADABLE", "ball_depth", "board_conditions",
    "board_layers", "board_row_notes", "condition_holds", "footprint_facts", "grid_misfit", "identities", "ladder_count", "name_tokens",
    "pad_spacing", "part_layers", "plan_for_ir",
]
