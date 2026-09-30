"""The ``kr447_tx_exciter`` template (``radio_build = tx_exciter``): the stage-4 bench board of the KR 447 MHz FM transmitter, conducted into 50 ohm only (kr447 design §2.4).

Invariant: a pure function of the confirmed requirements, the library on
disk and the block builders - nothing is guessed, nothing is grounded that is
not. The board is the composition (:func:`~ai_eda.design.rf.blocks.base.merge_results`)
of the transmit blocks of this part - :class:`~ai_eda.design.rf.blocks.tx_chain.TxModBlock`
(``tx_mod``, 8xx, the TCXO and the two PM tanks under a BMI-S-103 can),
:class:`~ai_eda.design.rf.blocks.tx_chain.TxChainBlock` (``tx_chain``, 8xx, the multipliers under the
BMI-S-105 can; the design's one can over both could not hold their parts,
see that module), :class:`~ai_eda.design.rf.blocks.tx_chain.TxDriverBlock`
(``tx_driver``, 8xx) and :class:`~ai_eda.design.rf.blocks.pa.PaBlock` (``pa``, 9xx, with
the harmonic low-pass and the U.FL output) - and their *companions*: the TX
power section, the PTT sequencer / interlock and the TX audio block of part
P9 (``PowerBlock(modes=("tx",))``, ``PttBlock(pa_stand_in=False, tot=...)``,
``TxAudioBlock(k_pm_key="tx.k_pm")``, kr447 design §2.1). The companions are
imported lazily by :func:`default_companions` (in the tree since the wave-2
merge); without them the template **refuses** ("the TX power, PTT and audio
blocks are not composed yet") instead of building a board without them;
:func:`bench_companions` - three bench headers standing in for them
(:class:`BenchTxBlock`: TX_5V / TX_3V3 / PA_5V / PA_PD and the audio
PM_DRIVE from the bench) - is what the tests compose
(``Kr447TxExciterTemplate(companions=bench_companions())``), a wiring honest
about being a stand-in (its plan note says so, and ``input_voltage`` /
``frequency_deviation`` / ``audio_bandwidth`` are served only by the P9
blocks). The build order is the data flow: the supply blocks first (they
share the rail levels ``power.tx_5v`` / ``power.tx_3v3``), then the
modulator (it writes the frequency plan and ``tx.k_pm``), the multipliers,
the driver and the PA, then the TX audio block (it designs its integrator
with ``tx.k_pm``).

Selection and the closed world (kr447 design §2.0): selected only by the
confirmed categorical requirement ``radio_build = tx_exciter``
(:meth:`triggered_by`); it needs ``carrier_frequency`` (a channel of the
unverified KR raster, else refused naming the channels), ``modulation``
(``fm``) and ``input_voltage``; it serves the keys of
``family.BUILDS["tx_exciter"]`` and refuses every other confirmed design
requirement with the family's sentence (``erp is served by
radio_build=transceiver or transceiver_conducted`` - no antenna on this
board). A stated ``pcb_layers`` other than 4 refuses before anything is asked
(the layer policy: the RF lines need a reference plane). A stated
``tx_power`` above the profile's ``kr447.max_power`` placeholder builds, and
``rf.regulatory_profile`` then FAILs (two confirmed values contradict each
other); one outside 0.01-1 W is refused (the PA stage's class).

What the plan carries: every ``kr447.*`` regulatory placeholder as a choice
whose description ends ``[UNVERIFIED: ...]`` (``ir.rf.profile_keys``;
``rf.regulatory_profile`` never PASSes), every block choice and ``model.*``
value (``ir.rf.model_values``: UNVERIFIED, ``rf.model_grounding``
NOT_VERIFIED), the floorplan regions (``floor.<block>.x`` / ``.y`` / ``.w``
/ ``.h`` mm, choices), the RF design ``ir.rf`` (blocks with ports, nine
fixture networks, the frequency plan's margin rows, lab items, the rail
budgets of the power block) and the signal-integrity classes
(:meth:`si_declarations`): ``RF50`` (the 50 ohm lines at f_c from the
band-pass output ``TX_RAW`` through the pads and the driver to the PA input,
target ``tx.z_mid``), ``RF50_H`` (the match output and the LPF input, which
carry the harmonics before the LPF: rf frequency 3 f_c, target ``rf.z0``),
``RF_OUT`` (the conducted output at f_c, target ``rf.z0``), ``TX_LUMPED``
(the tank, bias and PA-output nodes: lumped high-impedance nodes, no line
impedance, not marked RF) and ``POWER_PA`` (the TX supply path at the
IPC-2221 width for 0.6 A).

Expected statuses (measured 2026-09-29 with ngspice-42, the packed KiCad
10.0.6 libraries, no kicad-cli; ``tests/test_kr447_tx_exciter.py``): every
``spice.rf.*`` row of the ten fixture networks PASS as "a network verdict
under confirmed model values (not a measured part)" (41 results in about
0.9 s; the tenth, ``pa_lpf``, is the match and the low-pass as the board
joins them - the fixture-membership pass, part A1); the design deck's five
bias rows and three ``pm_couple_*`` rows PASS;
``rf.freq_plan``, ``calc.recompute`` and every ``block.interface.*`` PASS;
``rf.deviation`` NOT_VERIFIED on the bench stand-in (no integrator: no
``spice.pm_drive_peak``, no ``tx.tau_i``) and, with the P9 blocks and a
confirmed ``frequency_deviation``, PASS / FAIL on the factors this design
PASSed (2.1945 kHz against 2.5 kHz at the default models: PASS);
``rf.model_grounding`` / ``rf.regulatory_profile`` / ``rf.lab.*``
NOT_VERIFIED. The bench board places (106 x 42 mm, both cans with their parts
inside their fences; 106 x 73 mm with the P9 blocks) - the PA's QFN-12 with
its fan-out room, every other part 2.2 mm from its pads
(``placement.rf_floorplan`` 0.2, decision 2A) - and routes: ``routing.maze``
0.6 reads the PHA-1's ``SOT-89-3`` custom pad 2 as the boxes of its anchor and
primitives and joins the PA's fenced pads (U901.2 / .3 / .6 / .10 / .11) and
R806.2's ground pad to its grid by escapes; measured 2026-09-30 through the
pipeline (``tests/test_kr447_tx_exciter.py``): 57 of 57 nets in 19
negotiation iterations, PA_PD and TX_X3_E promoted and re-routed once,
``pcb.routing.connectivity`` NOT_VERIFIED (GND joined only through the plane
fill), ``domain.rf.impedance`` and the RF ``si.impedance.*`` PASS, and
``spice.si.PA_PD`` FAILs (the 89.73 mm line J2.2 - U901.10 overshoots 30.5 %
against its class's 15 %: an honest verdict on the copper, left to a human).
With the P9 blocks the MAX9814 (U301) gets its fan-out room too and the board
(108 nets, a large board: up to 150 negotiation iterations) is routing.maze
0.7's: 0.6's first pass left U301 pads without an escape (U302.2's ground via
in U301's escape area), the second run keeps U301's room clear of the other
footprints' plane vias and every U301 / U901 pad escapes; the router alone
connects 108 of 108 nets, legal after 93 iterations (2524 tracks, 467 vias,
4932.4 mm, about 480 s with four routes sharing 4 CPUs; at the old cap of 40
it had 4 conflicting nets left); through the whole pipeline (2026-09-30) six
nets are promoted to Z50 and the single re-route is legal after 36
iterations and kept (2392 tracks, 468 vias, 4872.8 mm), no check FAILs and
RELEASE is NOT_VERIFIED. ``si.rf_length``
is NOT_VERIFIED on any routed board too: ``ai_eda.design.board.add_board``
builds ``ir.si`` without ``rf_length_fraction`` (a wave-1 gap no template can
close). On the bench board ``review.requirements_vs_ir`` FAILs naming
``req.input_voltage`` only (the P9 power block serves it, not the headers).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ai_eda.ir import Block as TopologyBlock
from ai_eda.ir import CircuitDomain, CircuitIR, Constraint, ConstraintKind, MissingInformation, NetClass, NetKind, SimulationSetup, SpiceBinding, Stimulus, StimulusKind, Topology
from ai_eda.ir.rf import RFBlock, RFDesign, RFPort, RFRegion
from ai_eda.tools.calc import radio
from ai_eda.tools.calc.rf import chebyshev_attenuation_db
from ai_eda.tools.kicad.library import KicadLibrary

from ai_eda.design.base import (
    NO_RECORD,
    DesignChange,
    LayerPolicy,
    PartNote,
    Plan,
    Template,
    TheorySection,
    number,
    parameter_value,
    quantity,
    requirement_text,
    unserved_requirements,
    unverified,
)
from ai_eda.design.inputs import DesignInput, canonical_key, present_keys, read_modulation, read_radio_build
from ai_eda.design.library_parts import TemplateRefusal
from ai_eda.design.rf.blocks.base import Block, BlockBuilder, BlockContext, BlockPrefix, BlockResult, exclude_floating, merge_results
from ai_eda.design.rf.blocks.pa import P_OUT_RANGE_W, PaBlock, pa_net_classes
from ai_eda.design.rf.blocks.tx_chain import (
    CHAIN_NET_CLASSES,
    DRIVER_NET_CLASSES,
    K_PM_KEY,
    MOD_NET_CLASSES,
    TxChainBlock,
    TxDriverBlock,
    TxModBlock,
    calc,
    copy_input,
)
from ai_eda.design.rf.family import BUILDS, unserved_message
from ai_eda.design.rf.models import MODEL_VERDICT
from ai_eda.design.rf.parts import part_line
from ai_eda.design.rf.profile import PROFILE, PROFILE_BY_KEY, profile_choices, profile_keys, raster_refusal
from ai_eda.design.templates import THEORY_CURVE_NOTE, _changes, _curve_figure, _lin_grid, _missing_inputs, _net_line, _refused

if TYPE_CHECKING:
    from ai_eda.design.board import BoardContext, SIDeclarations
    from ai_eda.report.figures import Figure

#: the radio_build this template builds and the family's entry for it
RADIO_BUILD = "tx_exciter"
BUILD = BUILDS[RADIO_BUILD]
TEMPLATE_ID = BUILD.template_id
#: why only 4 layers (kr447 design §2.0, decision 7A)
PLANE_REASON = "the RF lines need a reference plane (decision 7A: 4 layers)"
#: the pack voltages the board is designed for (the stage-1 board's range: above the low-pack inhibit's release point, at most a full 2S pack)
V_IN_RANGE: tuple[float, float] = (6.6, 8.4)
#: the deviation / audio-bandwidth / time-out ranges the TX audio block is designed for (the stage-1 board's; Hz, Hz, s)
DEVIATION_RANGE: tuple[float, float] = (500.0, 5000.0)
BANDWIDTH_RANGE: tuple[float, float] = (2000.0, 4000.0)
TOT_RANGE_S: tuple[float, float] = (10.0, 600.0)
TX_TIMEOUT_KEY = "tx_timeout"
#: the parameter the TX audio block (part P9) holds the confirmed frequency_deviation in; the occupied-bandwidth rows read the same key
DEVIATION_PARAM = "tx.frequency_deviation"
#: requirement keys this board refuses because it has no antenna (the family's sentence is followed by this)
ANTENNA_KEYS: tuple[str, ...] = ("erp", "eirp", "antenna_impedance", "antenna_gain", "link_range", "field_strength_limit")
#: how each block of the board is numbered: references re-based by hundreds (the design's 1xx power, 2xx ptt, 3xx TX audio, 8xx TX chain, 9xx PA)
PREFIXES: dict[str, BlockPrefix] = {
    "tx_mod": BlockPrefix(800), "tx_chain": BlockPrefix(800), "tx_driver": BlockPrefix(800), "pa": BlockPrefix(900),
    "power": BlockPrefix(100), "ptt": BlockPrefix(200), "tx_audio": BlockPrefix(300), "bench_tx": BlockPrefix(0),
}
#: the floorplan region of each block (x, y, w, h mm from the board's top-left corner): choices ``floor.<block>.*``; the board is their bounding box.
#: The signal runs left to right along the top: the modulator's can, the multipliers' can, the driver above the PA, the output J901 at the right
#: edge; supply and audio (or the bench headers) below. A can's region holds its footprint's extent plus the placer's spacing.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "tx_mod": (0.0, 0.0, 31.0, 31.0),
    "tx_chain": (31.0, 0.0, 44.0, 31.0),
    "tx_driver": (75.0, 0.0, 31.0, 14.0),
    "pa": (75.0, 14.0, 31.0, 17.0),
    "bench_tx": (0.0, 31.0, 106.0, 11.0),
    "power": (0.0, 31.0, 42.0, 26.0),
    "tx_audio": (42.0, 31.0, 32.0, 36.0),
    "ptt": (74.0, 31.0, 32.0, 42.0),
}
_REGION_WHAT = {
    "tx_mod": "the modulator under its BMI-S-103 can (TCXO, PM tanks, followers, varactor bias) at the top left",
    "tx_chain": "the x3 / x2 / x2 multipliers and the output band-pass under the BMI-S-105 can, right of the modulator",
    "tx_driver": "the pads and the PHA-1 driver at the top right",
    "pa": "the PA, its match, the harmonic low-pass and the conducted U.FL output at the right edge",
    "bench_tx": "the bench headers along the bottom edge (stand-in for the P9 supply and audio blocks)",
    "power": "pack connector, main switch, TX rail switch and regulators at the bottom left",
    "tx_audio": "TX audio (microphone, limiter, splatter filter, integrator) at the bottom middle",
    "ptt": "PTT sequencer, TX interlock and PA supply switch at the bottom right",
}


# --------------------------------------------------------------------------- companions


class BenchTxBlock(Block):
    """Three bench headers in place of the P9 blocks: ``J1`` TX_5V / TX_3V3, ``J2`` PA_5V / PA_PD, ``J3`` PM_DRIVE (a stand-in, honest about it).

    The bench supplies ``TX_5V`` / ``TX_3V3`` / ``PA_5V`` (rail ports at
    ``power.tx_5v`` / ``power.tx_3v3``, the keys the TX blocks read), holds
    ``PA_PD`` low (PA on) and drives ``PM_DRIVE`` from an audio generator. In
    the design deck they are the ideal sources ``VTX5V`` / ``VTX3V3`` /
    ``VPA5V`` / ``VPAPD`` and ``VPMD`` (0 V DC, ac 1 V - the ``pm_couple_*``
    rows are ratios). It reads no requirement: ``input_voltage``,
    ``frequency_deviation``, ``audio_bandwidth`` and ``tx_timeout`` are served
    only once the P9 blocks replace it.
    """

    id = "bench_tx"
    title = "bench interface: TX_5V / TX_3V3 / PA_5V / PA_PD and PM_DRIVE from the bench (stand-in for the TX power, PTT and audio blocks)"
    interface_nets = ("TX_5V", "TX_3V3", "PA_5V", "PA_PD", "PM_DRIVE")

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        v5 = b.choice("power.tx_5v", 5.0, "V", "the bench supply on J1 for TX_5V (stand-in for the TX power section's LM1117DT-5.0 output)")
        v3 = b.choice("power.tx_3v3", 3.3, "V", "the bench supply on J1 for TX_3V3 (stand-in for the TX power section's LP5907-3.3 output)")
        pd = b.choice("bench.pa_pd_on", 0.0, "V", "the bench holds PA_PD low: the PA powered up (POWER_DOWN high = powered down [UNVERIFIED: NXP MMZ09332B datasheet])")
        ac = b.choice("bench.pm_ac", 1.0, "V", "the ac magnitude of the bench audio on PM_DRIVE (the pm_couple rows are ratios, so it cancels)")
        zero = b.choice("bench.pm_dc", 0.0, "V", "the bench audio's DC level on PM_DRIVE (the coupling capacitor blocks it)")
        j1 = b.part("header_3", "J1", "Conn_01x03", "bench header: TX_5V, TX_3V3, GND in")
        j2 = b.part("header_3", "J2", "Conn_01x03", "bench header: PA_5V, PA_PD, GND in (the PTT block's PA switch and power-down are not on this board)")
        j3 = b.part("header_3", "J3", "Conn_01x03", "bench header: PM_DRIVE audio in, GND, GND (the TX audio block is not on this board)")
        for ref in ("J1", "J2", "J3"):
            b.bind(ref, SpiceBinding(exclude=True, exclude_reason="bench header: a connector, no electrical model", provenance=ctx.provenance(f"{ref} excluded")))
        b.net("TX_5V", NetKind.POWER, j1.at("Pin_1"), "bench supply TX_5V")
        b.net("TX_3V3", NetKind.POWER, j1.at("Pin_2"), "bench supply TX_3V3")
        b.net("PA_5V", NetKind.POWER, j2.at("Pin_1"), "bench supply PA_5V")
        b.net("PA_PD", NetKind.SIGNAL, j2.at("Pin_2"), "bench PA power-down line")
        b.net("PM_DRIVE", NetKind.ANALOG, j3.at("Pin_1"), "bench audio into the phase modulator")
        b.net("GND", NetKind.GROUND, [*j1.at("Pin_3"), *j2.at("Pin_3"), *j3.at("Pin_2"), *j3.at("Pin_3")], "ground")
        for sid, net, value, note in (("VTX5V", "TX_5V", v5, "TX_5V from the bench"), ("VTX3V3", "TX_3V3", v3, "TX_3V3 from the bench"),
                                      ("VPA5V", "PA_5V", v5, "PA_5V from the bench (no PA supply switch on this board)"), ("VPAPD", "PA_PD", pd, "PA_PD held low by the bench")):
            b.result.stimuli.append(Stimulus(id=sid, source="voltage", net=net, reference_net="GND", kind=StimulusKind.DC, value=value, provenance=ctx.provenance(note)))
        b.result.stimuli.append(Stimulus(id="VPMD", source="voltage", net="PM_DRIVE", reference_net="GND", kind=StimulusKind.DC, value=zero, params={"ac": ac},
                                         provenance=ctx.provenance("PM_DRIVE driven by the bench audio generator (the TX audio block is not on this board)")))
        b.result.ports = [RFPort(name="TX_5V", net="TX_5V", kind="rail", voltage_v=v5, direction="out"),
                          RFPort(name="TX_3V3", net="TX_3V3", kind="rail", voltage_v=v3, direction="out"),
                          RFPort(name="PA_5V", net="PA_5V", kind="rail", voltage_v=v5, direction="out")]
        b.result.chain = ["J1", "J2", "J3"]
        b.result.notes.append("bench_tx: three headers stand in for the TX power section, the PTT sequencer / interlock and the TX audio block (part P9): "
                              "input_voltage, frequency_deviation, audio_bandwidth and tx_timeout are served by those blocks, not by this stand-in")
        return b.done()


@dataclass(frozen=True)
class Companions:
    """The blocks composed around the TX blocks: ``supply`` built first (they share the rails), ``audio`` last (it reads ``tx.k_pm``)."""

    supply: tuple[tuple[Block, BlockPrefix], ...]
    audio: tuple[tuple[Block, BlockPrefix], ...] = ()
    stand_in: bool = False


def bench_companions() -> Companions:
    """:class:`BenchTxBlock` alone (the tests' stand-in for the P9 blocks)."""
    return Companions(supply=((BenchTxBlock(), PREFIXES["bench_tx"]),), stand_in=True)


def default_companions(*, tot: bool) -> Companions | None:
    """The P9 TX power, PTT and audio blocks when they are in the tree (imported lazily), else ``None`` (the template then refuses)."""
    try:
        from ai_eda.design.rf.blocks.power import PowerBlock
        from ai_eda.design.rf.blocks.ptt import PttBlock
        from ai_eda.design.rf.blocks.tx_audio import TxAudioBlock
    except ImportError:
        return None
    return Companions(
        supply=((PowerBlock(modes=("tx",)), PREFIXES["power"]), (PttBlock(pa_stand_in=False, tot=tot), PREFIXES["ptt"])),
        audio=((TxAudioBlock(k_pm_key=K_PM_KEY), PREFIXES["tx_audio"]),),
    )


def tx_blocks() -> list[tuple[Block, BlockPrefix]]:
    """The four transmit blocks of this part with their prefixes, in build order (the transceiver composes the same builders)."""
    return [(TxModBlock(), PREFIXES["tx_mod"]), (TxChainBlock(), PREFIXES["tx_chain"]), (TxDriverBlock(), PREFIXES["tx_driver"]), (PaBlock(), PREFIXES["pa"])]


def tx_net_classes() -> dict[str, list[str]]:
    """The signal-integrity class members of the four transmit blocks (their ``net_classes`` merged; the exciter prefixes no net)."""
    out: dict[str, list[str]] = {}
    for classes in (MOD_NET_CLASSES, CHAIN_NET_CLASSES, DRIVER_NET_CLASSES, pa_net_classes()):
        for name, nets in classes.items():
            have = out.setdefault(name, [])
            have += [n for n in nets if n not in have]
    return out


def _board_values(ctx: BlockContext, block_ids: list[str]) -> BlockResult:
    """The rows that belong to the board, not to a block: the KR 447 MHz profile placeholders and the floorplan regions."""
    b = BlockBuilder(ctx, "board", "board values")
    for ch, traced in profile_choices(ctx.template_id, ctx.confirmed):
        b.result.choices.append(ch)
        b.result.params[ch.key] = traced
    for bid in block_ids:
        for axis, value in zip(("x", "y", "w", "h"), REGIONS[bid]):
            what = {"x": "left edge", "y": "top edge", "w": "width", "h": "height"}[axis]
            b.choice(f"floor.{bid}.{axis}", value, "mm", f"floorplan region of block {bid} ({_REGION_WHAT[bid]}): {what} (mm, board frame)")
    return b.done()


# --------------------------------------------------------------------------- the template


class Kr447TxExciterTemplate(Template):
    """``radio_build = tx_exciter``: TCXO -> PM tanks -> x12 -> BPF -> driver -> PA -> LPF -> U.FL, conducted only (module docstring)."""

    id = TEMPLATE_ID
    title = BUILD.title
    triggers = ("radio_build",)
    needs = BUILD.needs
    serves = BUILD.serves
    plane_nets = ("GND", None)
    layer_policy = LayerPolicy(allowed=(4,), default=4, reason=PLANE_REASON)

    def __init__(self, companions: Companions | None = None) -> None:
        self._companions = companions

    def companions(self, *, tot: bool) -> Companions | None:
        return default_companions(tot=tot) if self._companions is None else self._companions

    # ------------------------------------------------------------------ selection

    def triggered_by(self, ir: CircuitIR, inputs: dict[str, DesignInput]) -> bool:
        return read_radio_build(ir)[0] == RADIO_BUILD

    def refusals(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str]) -> list[MissingInformation]:
        """The family's closed world (a requirement another build serves names that build; no antenna here) and the FM-only rule."""
        out: list[MissingInformation] = []
        modulation, _why = read_modulation(ir)
        if modulation is not None and modulation != BUILD.modulation:
            why = f"modulation {modulation}: the KR 447 MHz licence-exempt class is FM (F3E) [UNVERIFIED: 「무선설비규칙」]; radio_build={RADIO_BUILD} builds an FM transmitter only"
            out.append(MissingInformation(key="modulation", required=False, rationale=why,
                                          question=f"{why}; no template design was proposed. State modulation=FM or choose another design."))
        for r in unserved_requirements(ir, self):
            canon = canonical_key(r.key) or r.key
            extra = " (no antenna on this board: its output is the conducted U.FL into an attenuator or a dummy load)" if canon in ANTENNA_KEYS else ""
            why = f"{r.id} ({requirement_text(r)}): {unserved_message(canon, RADIO_BUILD)}{extra}"
            out.append(MissingInformation(key=r.key, required=False, rationale=why,
                                          question=f"{why}; no template design was proposed. Start the project of the build that serves it, or leave this requirement out of the {RADIO_BUILD} board."))
        return out

    @staticmethod
    def _out_of_range(inputs: dict[str, DesignInput]) -> str | None:
        """A stated input outside what the board is designed for, or a carrier that is no channel of the (unverified) raster."""
        f = inputs.get("carrier_frequency")
        if f is not None:
            why = raster_refusal(float(f.traced.value), float(PROFILE_BY_KEY["kr447.band_low"].value), float(PROFILE_BY_KEY["kr447.band_high"].value),
                                 float(PROFILE_BY_KEY["kr447.channel_raster"].value))
            if why is not None:
                return f"{f.requirement.id}: {why}"
        checks = (
            ("input_voltage", V_IN_RANGE, "the 2S Li-ion pack above the confirmed cut-off 6.4 V plus the low-pack inhibit's 0.2 V hysteresis, at most a full pack"),
            ("frequency_deviation", DEVIATION_RANGE, "the limiter / integrator chain's design range (the KR profile's maximum is a separate check)"),
            ("audio_bandwidth", BANDWIDTH_RANGE, "the splatter filter's design range (a voice channel)"),
            (TX_TIMEOUT_KEY, TOT_RANGE_S, "the 4060 RC time-out's design range"),
            ("tx_power", P_OUT_RANGE_W, "the MMZ09332BT1 stage's class [UNVERIFIED: NXP MMZ09332B datasheet] (the profile's limit is a separate check)"),
        )
        for key, (lo, hi), why in checks:
            inp = inputs.get(key)
            if inp is not None and not lo <= float(inp.traced.value) <= hi:
                return f"{key} {float(inp.traced.value):.12g} {inp.traced.unit} ({inp.requirement.id}) is outside {lo:.12g}..{hi:.12g} {inp.traced.unit}: {why}"
        return None

    # ------------------------------------------------------------------ build

    def build(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str], library: KicadLibrary, *, confirmed: bool) -> Plan:
        t = self.id
        plan = Plan(template=t, title=self.title)
        stated = {k: inputs[k] for k in self.serves if k in inputs}
        why = self._out_of_range(inputs)
        if why is not None:
            plan.inputs = stated
            return _refused(plan, why)
        present = present_keys(ir, inputs)
        missing = [k for k in ("carrier_frequency", "input_voltage") if k not in present]
        if missing:
            _missing_inputs(plan, f"The {self.title} template", missing, unusable, examples={"carrier_frequency": "447.5625 MHz", "input_voltage": "7.4 V"})
        modulation, mod_why = read_modulation(ir)
        if modulation is None:
            plan.questions.append(MissingInformation(key="modulation", required=mod_why is None, rationale="template input",
                                                     question=mod_why or f"The {self.title} template needs modulation: answer modulation=fm (the KR 447 MHz class is FM [UNVERIFIED])"))
            if not missing:
                return _refused(plan, f"modulation missing{': ' + mod_why if mod_why else ''}")
        if plan.questions or plan.notes:
            return plan
        if modulation != BUILD.modulation:
            return _refused(plan, f"modulation {modulation!r}: the KR 447 MHz licence-exempt class is FM telephony [UNVERIFIED: 「무선설비규칙」]; this board transmits FM only")
        tot = TX_TIMEOUT_KEY in inputs
        comp = self.companions(tot=tot)
        if comp is None:
            return _refused(plan, ("the TX power section, the PTT sequencer / interlock and the TX audio block (kr447 design §2.1, part P9) are not composed yet: "
                                   "this board is the modulator, the multipliers, the driver and the PA plus those three blocks, and a board without them would leave "
                                   "input_voltage, frequency_deviation and audio_bandwidth unserved (default_companions() imports them once they are in the tree)"))
        plan.inputs = stated
        profile = profile_choices(t, confirmed)
        ctx = BlockContext(ir=ir, library=library, template_id=t, confirmed=confirmed, inputs=inputs, shared={c.key: tr for c, tr in profile})
        results: list[BlockResult] = []
        try:
            for blk, prefix in (*comp.supply, *tx_blocks(), *comp.audio):
                if blk.id not in REGIONS:
                    raise TemplateRefusal(f"block {blk.id!r} has no floorplan region in REGIONS")
                r = blk.build(ctx, prefix)
                results.append(r)
                ctx.shared.update({k: v for k, v in r.params.items() if k not in ctx.shared})
            board_values = _board_values(ctx, [r.block_id for r in results])
            self._obw_rows(ctx, board_values)
            merged = merge_results([board_values, *results], t)
        except TemplateRefusal as e:
            return _refused(plan, str(e))
        components, report = exclude_floating(merged.components, merged.nets, merged.stimuli, t)
        bad = [f for f in report.unresolved if not f.ref.startswith("RV")]
        if bad:
            return _refused(plan, "a multi-terminal part has a node nothing else simulated touches: " + "; ".join(f.reason for f in bad))
        rf_blocks = [RFBlock(id=r.block_id, title=r.title, refs=[c.ref for c in r.components], chain=list(r.chain), shield_ref=r.shield_ref,
                             region=self._region(merged, r.block_id), ports=list(r.ports)) for r in results]
        try:
            rf = RFDesign(blocks=rf_blocks, networks=list(merged.networks), frequency_plan=list(merged.plan_lines), lab_items=list(merged.lab_items),
                          rails=list(merged.rails), model_values=sorted(merged.model_keys), profile_keys=profile_keys())
        except ValueError as e:
            return _refused(plan, f"the RF design refuses the composition: {e}")
        plan.choices = list(merged.choices)
        plan.computed = list(merged.computed)
        plan.parts = [part_line(merged.placed[c.ref]) for c in components if c.ref in merged.placed]
        plan.nets = [_net_line(n) for n in merged.nets]
        sim = SimulationSetup(stimuli=list(merged.stimuli), analyses=list(merged.analyses), expectations=list(merged.expectations))
        plan.simulation = self._simulation_lines(merged, sim, components, report)
        plan.notes.extend(merged.notes)
        topology = Topology(
            name=self.title, domains=[CircuitDomain.RF, CircuitDomain.ANALOG, CircuitDomain.POWER],
            rationale=("KR 447 MHz FM transmitter exciter, conducted into 50 ohm only: TCXO f_c / 12 -> two buffered varactor phase-modulator tanks on one "
                       "bias node (the integrated audio PM_DRIVE: indirect FM) -> x3 / x2 BFR92 multipliers with double-tuned top-C tanks -> x2 into a 5-resonator "
                       "top-C band-pass (the last tank and the band-pass as one network) -> pad -> PHA-1 driver -> pad -> MMZ09332BT1 PA -> load-line L-match -> 7th-order Chebyshev low-pass -> U.FL. "
                       "No IC has a model: the passive networks are the RF fixtures, judged under confirmed model values"),
            provenance=ctx.provenance(f"selected by radio_build = {RADIO_BUILD}"),
            blocks=[TopologyBlock(id=r.block_id, function=r.title, domain=CircuitDomain.RF if r.block_id in ("tx_mod", "tx_chain", "tx_driver", "pa") else CircuitDomain.ANALOG,
                                  component_refs=[c.ref for c in r.components], provenance=ctx.provenance(f"block {r.block_id}")) for r in results],
        )
        plan.changes = _changes(t, self.title, topology, components, merged.nets, dict(merged.params), sim, self._constraints(ctx))
        plan.changes.append(DesignChange(description="RF design: blocks, fixture networks, frequency plan, lab items, rails", target="rf", operation="set", payload=rf,
                                         rationale=f"template {t}: {len(rf.networks)} fixture network(s), {len(rf.frequency_plan)} plan row(s), {len(rf.lab_items)} lab item(s)"))
        return plan

    # ------------------------------------------------------------------ build steps

    @staticmethod
    def _obw_rows(ctx: BlockContext, b_result: BlockResult) -> None:
        """The single-tone 99 % occupied bandwidth at 1 kHz and 3 kHz of the stated deviation (else the profile's), ``calc.rf.fm.obw99`` - numbers, no verdict."""
        b = BlockBuilder(ctx, "board", "board values")
        b.result = b_result
        dev_in = ctx.inputs.get("frequency_deviation")
        if dev_in is not None:  # the confirmed requirement under the TX audio block's key (its copy when it is composed)
            dev_key, dev = DEVIATION_PARAM, copy_input(b, DEVIATION_PARAM, dev_in.traced)
        else:
            dev_key, dev = "kr447.max_deviation", b.result.params["kr447.max_deviation"]
        for tag, f in (("1k", 1000.0), ("3k", 3000.0)):
            fm = b.choice(f"tx.obw.f_{tag}", f, "Hz", f"the single test tone of the {tag} occupied-bandwidth number (which tone the KR test method prescribes is [UNVERIFIED])")
            calc(b, f"tx.obw99_{tag}", lambda fm=fm, tag=tag: radio.fm_obw99(dev, fm, (dev_key, f"tx.obw.f_{tag}")))

    @staticmethod
    def _region(merged: BlockResult, bid: str) -> RFRegion | None:
        vals = {axis: merged.params.get(f"floor.{bid}.{axis}") for axis in ("x", "y", "w", "h")}
        if any(v is None for v in vals.values()):
            return None
        return RFRegion(**vals)  # type: ignore[arg-type]

    @staticmethod
    def _simulation_lines(merged: BlockResult, sim: SimulationSetup, components, report) -> list[str]:
        lines = []
        for nw in merged.networks:
            rows = ", ".join(f"{e.id}{'[' + e.state + ']' if e.state else ''} ({e.quantity} at {float(e.at.value):.9g} Hz: "
                             + (f"{e.bound} {float(e.nominal.value):.6g}" if e.bound else f"{float(e.nominal.value):.6g} +/- {float(e.tol_abs.value):.6g}") + ")"  # type: ignore[union-attr]
                             for e in nw.expectations)
            lines.append(f"fixture {nw.id}: {len(nw.members)} member(s) between {', '.join(f'{p.name} ({p.kind})' for p in nw.ports)}"
                         + (f", states {', '.join(s.id for s in nw.states)}" if nw.states else "") + f"; rows {rows}")
        lines.append(f"every fixture PASS is a {MODEL_VERDICT}, schematic level (no track, via or ground-return inductance)")
        lines += [f"stimulus {s.id} ({s.kind.value}) on {s.net}: {s.provenance.note}" for s in sim.stimuli]
        for a in sim.analyses:
            params = ", ".join(f"{k} {v.value!r}" for k, v in a.params.items())
            lines.append(f"analysis {a.id} ({a.kind.value}{': ' + params if params else ''})")
        for e in sim.expectations:
            rule = f"{e.bound} {e.nominal.value:.6g}" if e.bound else f"{e.nominal.value:.6g} +/- " + (f"{e.tol_abs.value:.3g}" if e.tol_abs else f"{e.tol_rel.value:.0%}")  # type: ignore[union-attr]
            at = f" at {e.at.value:.6g}" if e.at is not None else ""
            ref = f" re {e.reference_vector}" if e.reference_vector else ""
            lines.append(f"expectation {e.id}: {e.vector}{ref} {e.reduce.value}{at} {rule} {e.nominal.unit or ''} ({e.provenance.note})")
        excluded = [c.ref for c in components if c.spice is not None and c.spice.exclude]
        lines.append(f"excluded from the design deck ({len(excluded)}): {', '.join(excluded)} (reasons in each part's SPICE binding; "
                     f"{len(report.excluded)} of them dead branches found by exclude_floating)")
        lines += [f"reported, kept: {f.reason}" for f in report.unresolved]
        return lines

    @staticmethod
    def _constraints(ctx: BlockContext) -> list[Constraint]:
        s = ctx.provenance
        return [
            Constraint(id="c.kr447.conducted_only", kind=ConstraintKind.REGULATORY, target="*", provenance=s("stage-4 conducted exciter"),
                       description=("stage 4 of the KR 447 MHz family: the exciter transmits only conducted, through the U.FL output into an attenuator or a "
                                    "dummy load; it never gets an antenna. Every kr447.* number is an UNVERIFIED placeholder; KC conformity assessment "
                                    "(전파법 제58조의2) is needed before any transmission, a self-built unit included, and whether a conducted bench test needs a "
                                    "permit is [UNVERIFIED: 전파법 제58조의3; ask 국립전파연구원 or the test lab]")),
            Constraint(id="c.kr447.rf_ground_vias", kind=ConstraintKind.RF, target="*", provenance=s("stopband of the fixtures"),
                       description=("every shunt part of the tanks, the band-pass, the match and the harmonic low-pass needs a via to the ground plane at its "
                                    "pad: at 447 MHz 1 mm of track to ground is about 1 nH, and the fixtures (no ground-return inductance) hold only then")),
            Constraint(id="c.kr447.pa_pd_polarity", kind=ConstraintKind.ELECTRICAL, target="PA_PD", provenance=s("PA power-down polarity"),
                       description=("PA_PD high = PA powered down (from the library pin name POWER_DOWN [UNVERIFIED: NXP MMZ09332B datasheet]); check the "
                                    "polarity before the first key-up")),
        ]

    # ------------------------------------------------------------------ board

    #: the TX supply path the POWER_PA class sizes (0.6 A, the transceiver's TX budget with margin)
    POWER_PA_CURRENT_A = 0.6
    POWER_PA_TEMP_RISE_C = 10.0

    def si_declarations(self, ir: CircuitIR, ctx: BoardContext) -> SIDeclarations:
        """``RF50`` / ``RF50_H`` / ``RF_OUT`` (the 50 ohm lines), ``TX_LUMPED`` (lumped nodes, no target) and ``POWER_PA`` (IPC-2221 width)."""
        from ai_eda.design.board import SIDeclarations
        from ai_eda.tools.calc.basic import ipc2221_width_for_current

        out = SIDeclarations()
        z_mid, z0, f_c, f_3 = (ctx.params.get(k) for k in ("tx.z_mid", "rf.z0", "rf.f_c", "lpf.f_3"))
        if None in (z_mid, z0, f_c, f_3):
            return out
        tol = ctx.choice("si.rf50_z0_tol", 0.10, None, "tolerance of the RF50 / RF50_H / RF_OUT classes' impedance (10 %)")
        prov = ctx.structural
        nets = tx_net_classes()
        out.classes += [
            NetClass(name="RF50", nets=nets["RF50"], target_z0_ohm=z_mid, z0_tol_rel=tol, rf_frequency_hz=f_c,
                     description="the 50 ohm lines from the band-pass output TX_RAW through the pads and the driver to the PA input, over the In1.Cu ground plane",
                     provenance=prov("TX 50 ohm lines at tx.z_mid")),
            NetClass(name="RF50_H", nets=nets["RF50_H"], target_z0_ohm=z0, z0_tol_rel=tol, rf_frequency_hz=f_3,
                     description="the match output and the low-pass input: they carry the PA's harmonics before the low-pass (rf frequency 3 f_c: lambda_g / 10 is about 12 mm)",
                     provenance=prov("harmonic-carrying lines before the low-pass")),
            NetClass(name="RF_OUT", nets=nets["RF_OUT"], target_z0_ohm=z0, z0_tol_rel=tol, rf_frequency_hz=f_c,
                     description="the conducted output to the U.FL at the system impedance", provenance=prov("conducted output line")),
            NetClass(name="TX_LUMPED", nets=nets["TX_LUMPED"],
                     description=("the PM tanks, followers, multiplier collectors / bases, tank resonators, band-pass resonators, the PA output at its 20 ohm "
                                  "load line and the low-pass's inner nodes: lumped high-impedance (or low-impedance) nodes, no line impedance applies - keep "
                                  "them short; not marked RF (no target Z0 exists to judge)"),
                     provenance=prov("TX lumped nodes")),
        ]
        i = ctx.choice("si.power_pa_current", self.POWER_PA_CURRENT_A, "A", (
            "current the TX supply path's minimum width is sized for: 0.6 A, above the transceiver's 441 mA TX budget [UNVERIFIED: the budget is an estimate]"), param=True)
        dt = ctx.choice("si.power_pa_temp_rise", self.POWER_PA_TEMP_RISE_C, "degC", "temperature rise the POWER_PA class's IPC-2221 width is sized for", param=True)
        if ctx.stackup is not None:
            layer = ctx.stackup.copper_layer("F.Cu")
            if layer is not None:
                w = ctx.computed_param("si.w_power_pa", ipc2221_width_for_current(i, dt, layer.thickness_um,
                                                                                    ("si.power_pa_current", "si.power_pa_temp_rise", "pcb.stackup.copper[F.Cu].thickness_um")))
                nets = ["TX_5V", "PA_5V"] + (["VBAT", "VBAT_F", "V_SYS", "V_TX"] if "power.v_in" in ctx.params else [])
                out.classes.append(NetClass(name="POWER_PA", nets=nets, min_width_mm=w, power_current_a=i, power_temp_rise_c=dt,
                                            description="the TX supply path to the PA and the driver: at least the IPC-2221 width for 0.6 A",
                                            provenance=prov("TX supply path minimum width")))
        out.lines.append("RF50 / RF50_H / RF_OUT: the 50 ohm lines; TX_LUMPED: lumped nodes without a line impedance; si.rf_length needs ir.si.rf_length_fraction, "
                         "which the board module does not write yet")
        return out

    # ------------------------------------------------------------------ report views

    def _p(self, ir: CircuitIR, key: str) -> float | None:
        return parameter_value(ir, key)

    def theory(self, ir: CircuitIR) -> list[TheorySection]:
        p = lambda k: self._p(ir, k)  # noqa: E731
        q = lambda k, u: quantity(p(k), u)  # noqa: E731
        f_c, f_t, n_mult = p("rf.f_c"), p("tx.f_ref"), p("rf.n_mult")
        stab, f_err = p("rf.frequency_tolerance") or p("rf.tcxo_stability"), p("tx.f_error")
        profile_rows = "\n".join(f"| `{e.key}` | {e.value if isinstance(e.value, str) else quantity(float(e.value), e.unit)} | {e.document} |" for e in PROFILE)
        phases = {(k, tag): p(f"pm_mod{k}.phase.{tag}") for k in (1, 2) for tag in ("lo", "nom", "hi")}
        v_lo, v_nom, v_hi = p("pm.v_lo"), p("pm.v_bias"), p("pm.v_hi")
        chord = None
        if None not in (phases[(1, "lo")], phases[(1, "hi")], v_lo, v_hi) and v_hi != v_lo:  # type: ignore[operator]
            chord = math.radians(phases[(1, "hi")] - phases[(1, "lo")]) / (v_hi - v_lo)  # type: ignore[operator]
        lin = None
        if None not in (phases[(1, "lo")], phases[(1, "hi")], phases[(1, "nom")]) and phases[(1, "hi")] != phases[(1, "lo")]:
            lin = (phases[(1, "hi")] + phases[(1, "lo")] - 2 * phases[(1, "nom")]) / (phases[(1, "hi")] - phases[(1, "lo")])  # type: ignore[operator]
        tank_rows = []
        for k, mult, nxt in ((1, "x3", "x6"), (2, "x2", "x12")):
            tank_rows.append(f"| tx_tank{k} ({mult}) | {q(f'tx.f{k}', 'Hz')} | {q(f'tx.tank{k}.l', 'H')} | {q(f'tx.tank{k}.bw', 'Hz')} | "
                             f"{q(f'tx_tank{k}.port_r', 'ohm')} + j{q(f'tx_tank{k}.port_x', 'ohm')} / {q(f'tx.tank{k}.r_load_eff', 'ohm')} | "
                             f"{q(f'tx_tank{k}.c_tap_in', 'F')} / {q(f'tx_tank{k}.c_tap_out', 'F')} | {q(f'tx_tank{k}.c_couple.1', 'F')} | "
                             f"{q(f'tx_tank{k}.c_shunt.1', 'F')} / {q(f'tx_tank{k}.c_shunt.2', 'F')} | {number(p(f'tx_tank{k}.s21'), 4)} dB "
                             f"(Cohn {number(_neg(p(f'tx_tank{k}.loss')), 4)} dB) | {number(p(f'tx_tank{k}.rel_m'), 4)} / {number(p(f'tx_tank{k}.rel_p'), 4)} dB |")
        lpf_rows = " / ".join(quantity(p(f"lpf.{'c' if k % 2 else 'l'}.{k}"), "F" if k % 2 else "H") for k in range(1, 8))
        g_rows = ", ".join(number(p(f"lpf.g.{k}"), 5) for k in range(1, 8))
        att2 = att3 = None
        if None not in (f_c, p("lpf.f_edge"), p("lpf.ripple")):
            try:
                att2 = chebyshev_attenuation_db(7, p("lpf.ripple"), 2 * f_c, p("lpf.f_edge"))  # type: ignore[arg-type,operator]
                att3 = chebyshev_attenuation_db(7, p("lpf.ripple"), 3 * f_c, p("lpf.f_edge"))  # type: ignore[arg-type,operator]
            except ValueError:
                pass
        r_l, z0 = p("pa.r_l"), p("rf.z0")
        q_m = math.sqrt(z0 / r_l - 1.0) if r_l and z0 and z0 > r_l else None
        k_pm, k_tot = p("pm.k_pm"), p("tx.k_pm")
        return [
            TheorySection("개요: 전도 출력 송신 여진기 시험 보드", (
                "KR 447 MHz 면허 불요 FM 무전기 계열의 4단계 기판입니다. 신호 흐름: TCXO(f_c/12) → 버퍼로 분리된 바랙터 위상 변조 탱크 두 개(적분된 음성 "
                "PM_DRIVE 로 간접 FM) → BFR92 체배기 ×3 → ×2 (단마다 이중 동조 상단 결합 탱크) → ×2 (콜렉터에서 바로 5공진기 상단 결합 출력 대역통과 필터) → 패드 → "
                "PHA-1 드라이버 → 패드 → MMZ09332BT1 전력 증폭기 → 부하선 L 정합 → 7차 체비셰프 저역통과 필터 → U.FL 출력.\n\n"
                "차폐: 변조기(TCXO, 위상 변조 탱크 두 개, 팔로워, 바랙터 바이어스)는 BMI-S-103 캔, 체배기 세 단은 BMI-S-105 캔 아래에 있습니다. 설계서는 "
                "BMI-S-105 하나로 둘 다 덮었지만, 그 펜스 안(배치기의 1 mm 링을 뺀 34.9 × 22.2 mm)에는 두 블록 부품 82 개 중 75 개만 들어가므로(부품 사이 "
                "1 mm) 캔을 둘로 나눴습니다.\n\n"
                "**이 기판은 전도(conducted) 출력만 합니다.** 출력은 U.FL 로 감쇠기나 더미 로드에 연결하며 안테나를 달지 않습니다. 모든 KR 규제 수치(`kr447.*`)는 "
                "법령 원문으로 근거를 두지 않은 자리표시값(UNVERIFIED)이고 `rf.regulatory_profile` 은 PASS 가 되지 않습니다. 자작품이라도 송신 전에 KC 적합성평가"
                "(전파법 제58조의2)가 필요하며, 더미 로드로의 전도 시험에 허가가 필요한지도 검증되지 않았습니다(전파법 제58조의3 확인 필요).\n\n"
                "IC(TCXO, PHA-1, PA, 전원·PTT·오디오의 IC)는 SPICE 모델이 없어 모두 넷리스트에서 제외됩니다. 대신 그 사이의 수동 회로망 9 개(위상 변조 탱크 2, "
                "체배 탱크 2, 출력 대역통과 필터, 패드 2, 부하선 정합, 저역통과 필터)를 RF 픽스처로 잘라 ngspice 교류 해석으로 판정합니다. 체배 단의 픽스처에는 "
                "그 단의 콜렉터 초크(와 0 Ω 링크)와 다음 단의 베이스 분압기가 포함됩니다: 포트 모델 옆에 실제로 달린 부품이기 때문입니다. 그 PASS 는 '확인된 모델값 "
                "아래에서의 회로망 판정(측정된 부품이 아님)'이며 트랙·비아·접지 귀환 인덕턴스가 없는 회로도 수준입니다.\n\n"
                "| 규제 프로파일 키 | 값 (검증되지 않음) | 확인할 문서 |\n|---|---|---|\n" + profile_rows
            )),
            TheorySection("주파수 계획 (결정 1B: N = 12)", (
                f"- 반송파 f_c = {quantity(f_c, 'Hz')} (확인된 요구사항; 검증되지 않은 12.5 kHz 래스터의 채널이어야 함).\n"
                f"- 체배수 N = {number(n_mult)} = 3 × 2 × 2, 기준 f_T = f_c / N = {q('tx.f_ref', 'Hz')} (`calc.clock.divided`; KT2520K-T 라이브러리 범위 "
                "'10-60MHz' 안의 맞춤 주파수 - 채널을 원문으로 확인하기 전에는 주문하지 않음).\n"
                f"- 단 출력: f_1 = 3 f_T = {q('tx.f1', 'Hz')}, f_2 = 6 f_T = {q('tx.f2', 'Hz')}, f_3 = 12 f_T = {q('tx.f3', 'Hz')} (`calc.rf.mult.stage`, "
                "마지막 단이 정확히 f_c 에 떨어지지 않으면 템플릿이 거부).\n"
                f"- 가장 가까운 체배 스퓨리어스: f_c ∓ f_T = {q('tx.bpf.f_m1', 'Hz')} / {q('tx.bpf.f_p1', 'Hz')} (`calc.rf.mult.spur`), ∓ 2 f_T = "
                f"{q('tx.bpf.f_m2', 'Hz')} / {q('tx.bpf.f_p2', 'Hz')}, ∓ 3 f_T = {q('tx.bpf.f_m3', 'Hz')} / {q('tx.bpf.f_p3', 'Hz')}; 잔류 f_c/2 = "
                f"{q('tx.f2', 'Hz')}, 3/2 f_c = {q('tx.f_3half', 'Hz')}. 이들이 대역통과 필터의 통과대역 밖(대역폭 {q('tx.bpf.bw', 'Hz')} 이상 떨어짐)에 있다는 것은 "
                "`rf.freq_plan` 의 여유 행(산술)이고, 출력에서의 레벨은 실험 항목 `rf.lab.tx_spurious` 입니다.\n"
                f"- TCXO 안정도 {number(stab)} ppm 은 반송파에서 {quantity(f_err, 'Hz')} (`calc.rf.ppm_offset`; 데이터시트 값은 검증되지 않음)."
            )),
            TheorySection("위상 변조기: 버퍼로 분리된 탱크 두 개", (
                "각 탱크: 소스 포트(R_port) → DC 차단 C_dc → 직렬 저항 R_s → 탱크 노드(C_fixed + 트리머 + 바랙터 → GND, L → 바이어스 노드 VAR_B). VAR_B 는 "
                "바이패스 C_bypass 로 RF 접지되고 R_feed 를 거쳐 분압 노드 PM_BIAS 에서 바이어스를 받습니다.\n\n"
                "    L = 1/(ω_T² C_tot)                         (`calc.rf.lc.l_for_resonance`)\n"
                "    C_var(V) = CJO / (1 + V/VJ)^M              (`calc.rf.varactor.c_at_bias`)\n"
                "    C_fixed = C_tot − C_var(V0) − C_trim       (`calc.rf.pm.c_fixed`)\n"
                "    R_s = 1/(1/(Q_L ωL) − 1/(Q_u ωL) − 1/R_load) − R_port   (`calc.rf.pm.source_r_loaded`)\n"
                "    K_pm ≈ Q_L |dC/dV| / C_tot                 (`calc.rf.pm.k_pm`, 탱크당 소신호 기울기)\n\n"
                f"| 항목 | 값 |\n|---|---|\n| f_T | {q('tx.f_ref', 'Hz')} |\n| C_tot / Q_L | {q('pm.c_tot', 'F')} / {number(p('pm.q_l'))} |\n"
                f"| L | {q('pm.l', 'H')} |\n| C_var(V0 = {quantity(v_nom, 'V')}) | {q('pm.c_var.nom', 'F')} |\n| C_trim (중간 위치) / C_fixed | {q('pm.c_trim', 'F')} / {q('pm.c_fixed', 'F')} |\n"
                f"| R_s (탱크 1 / 2) | {q('pm.r_s1', 'ohm')} / {q('pm.r_s2', 'ohm')} |\n| 분압 R_top / R_bottom (V0 설정) | {q('pm.r_div_top', 'ohm')} / {q('pm.r_div_bottom', 'ohm')} |\n"
                f"| K_pm (탱크당, 설계값) | {quantity(k_pm, 'rad/V')} |\n\n"
                "설계서는 탱크 사이에 이미터 팔로워 하나와 x3 단으로의 '약한 결합(선택값)'을 두었습니다. 등록된 계산기 `calc.rf.pm.tank_phase_loaded` 가 탱크 노드의 "
                "부하로 모델링할 수 있는 것은 저항 하나뿐이므로, 두 번째 탱크 뒤에도 같은 이미터 팔로워(MMBT3904)를 두어 두 탱크가 같은 설계 회로망"
                "(소스 50 Ω, 부하 `model.buf.r_in`)이 되게 했습니다. 그래서 픽스처가 측정하는 것이 계산기가 계산하는 회로망과 정확히 같습니다. "
                "팔로워의 베이스 분압기와 첫 팔로워의 이미터 저항은 픽스처 포트 네트에 있지만 구성원이 아니라 포트 모델 안에 있습니다: `model.buf.r_in` 은 "
                f"분압기({q('tx.buf1.r_b1', 'ohm')} ∥ {q('tx.buf1.r_b2', 'ohm')})를 포함한 팔로워 입력이라 그 병렬 저항을 넘을 수 없고, `model.buf.r_out` 은 "
                f"이미터 저항({q('tx.buf1.r_e', 'ohm')})을 포함한 출력입니다. 모델값 설명이 그렇게 말하고, 픽스처의 포트 저항이 그 설명을 출처로 지니며, "
                "`rf.model_grounding` 이 그 행을 보여 줍니다. 두 탱크의 인덕터는 같은 바이패스 노드 VAR_B 로 돌아가므로, 각 픽스처에는 다른 탱크의 인덕터가 "
                "빠져 있습니다. 그 인덕터는 다른 탱크의 병렬 커패시턴스와 f_T 에서 공진하므로 VAR_B 에서 본 가지는 리액턴스(142 Ω)가 아니라 약 14–15 Ω 의 "
                "직렬 공진이고, 바이패스(0.43 Ω)는 그 약 3 % 입니다(기본 선택값). 수동 부하로 달면 각 픽스처의 위상이 최대 0.032° 움직입니다(결합 회로의 "
                "선형 노드 해석, ngspice-42 도 같은 값). 그러나 기판에서는 첫 팔로워가 탱크 1 의 신호로 탱크 2 를 구동하므로 공유 바이패스에 두 탱크의 인덕터 "
                "전류가 함께 흐릅니다. 팔로워를 `model.buf.r_out` 뒤의 이득 1 전압원으로 둔 선형 추정으로 탱크 1 의 위상은 +0.70–+1.04°, 탱크 2 의 위상은 "
                "자기 구동 대비 모든 상태에서 +2.85° 움직입니다. 트리머로 정렬하는 중심 어긋남이며, 두 탱크를 합친 현(φ_hi − φ_lo)은 79.4° 중 약 0.03° 만 "
                "변합니다. 따라서 픽스처의 절대 `phase21_deg` 행은 탱크 하나에 대한 판정이고 기판에서 결합된 두 탱크의 판정이 아니며, 결합 회로는 실험 항목 "
                "`tx_pm_coupling` 이 맡습니다. 탱크마다 바이어스·바이패스 노드를 따로 두면 결합이 없어지지만 설계 변경이라 사람이 정해야 하고, "
                "`pm.c_bypass` 를 키우면 `pm.r_feed` 와의 저역 통과가 음성 대역으로 들어옵니다.\n\n"
                "**정확한 회로망 위상.** 이상적인 탱크 식 φ = −atan(Q(f/f0 − f0/f)) 는 이 회로망에서 0.8–1.4° 어긋나 허용오차 1° 로는 올바른 탱크를 FAIL 시킵니다"
                "(critic2). 그래서 각 상태의 공칭값은 포트 R, DC 차단, R_s, 직렬 손실이 있는 L, 바이패스와 급전 저항, 부하를 모두 포함한 정확한 위상입니다:\n\n"
                f"| 상태 | 바이어스 | 위상 (공칭, 탱크 1 = 탱크 2) |\n|---|---|---|\n| bias_lo | {quantity(v_lo, 'V')} | {number(phases[(1, 'lo')], 5)} ° |\n"
                f"| bias_nom | {quantity(v_nom, 'V')} | {number(phases[(1, 'nom')], 5)} ° |\n| bias_hi | {quantity(v_hi, 'V')} | {number(phases[(1, 'hi')], 5)} ° |\n\n"
                f"bias_nom 의 {number(phases[(1, 'nom')], 3)}° 는 바이패스와 L 이 직렬로 만드는 탱크 자체의 중심 어긋남으로, 트리머로 정렬합니다. 현(chord) 기울기 "
                f"(φ_hi − φ_lo)/(V_hi − V_lo) = {number(chord, 5)} rad/V, 선형성 지표 (φ_hi + φ_lo − 2φ_nom)/(φ_hi − φ_lo) = {number(lin, 4)} (판정 없음, 기록만)."
            )),
            TheorySection("간접 FM 과 주파수 편이", (
                "위상 변조기 앞에서 제한된 음성을 적분하면 위상 변조가 주파수 변조가 됩니다(Armstrong). 편이는\n\n"
                "    Δf = N · K_pm · a · V_max / (2π · τ_i)        (`rf.deviation`)\n\n"
                f"여기서 K_pm = K_pm1 + K_pm2 (한 바이어스 노드의 두 탱크는 더해짐). `rf.deviation` 은 설계값이 아니라 픽스처 pm_mod1 / pm_mod2 의 bias_lo / bias_hi "
                "위상에서 잰 현 기울기를 합해 곱합니다. 송신 오디오 블록의 적분기는 설계 상수 `tx.k_pm` = 2 × K_pm(탱크당) = "
                f"{quantity(k_tot, 'rad/V')} 로 설계됩니다(두 기울기를 더하는 등록 계산기가 없어 이 합은 선택값으로 표에 보임).\n\n"
                "계수 a = |v(VAR_B)/v(PM_DRIVE)| 는 C_audio 와 분압기 테브난 저항의 고역통과, R_feed 와 C_bypass 의 저역통과의 곱입니다:\n\n"
                "    a(f) = −10·log10(1 + (f_hp/f)²) − 10·log10(1 + (f/f_lp)²)   [dB]   (`calc.audio.highpass1.db_at` + `.lowpass1.db_at`)\n\n"
                f"f_hp = {q('pm.f_hp', 'Hz')}, f_lp = {q('pm.f_lp', 'Hz')} (두 모서리 모두 음성 대역 밖), a(1 kHz) = {number(p('pm.couple.a_ref'), 4)} dB. 설계 덱이 "
                "300 Hz / 1 kHz / 3 kHz 에서 a 를 재어 1 kHz 공칭값과 비교합니다(`pm_couple_*`): 10 kΩ 급전이면 저역 모서리가 1.59 kHz 가 되어 편이가 평탄하지 "
                "않으므로(critic2) 1 kΩ 급전을 택했고, 바이패스는 탱크 중심 어긋남을 정하므로 크게 유지했습니다. 실제 편이는 실험 항목 `rf.lab.tx_deviation` 입니다."
            )),
            TheorySection("체배기와 이중 동조 탱크", (
                "각 단: BFR92 (분압 바이어스, R_E // C_E, 0 Ω 링크와 디커플링, 콜렉터 초크) 뒤에 상단 결합(top-C) 2공진기 탱크. 끝단 외부 Q_e = "
                f"{number(p('tx.tank_qe'))} 에서 BW = g_1 f0 / Q_e (`calc.rf.resonator.top_c.bw_for_qe`), 공진 C_res = 1/(ω0² L), 결합 C_k = k (BW/f0) C_res, "
                "용량성 탭 C_s = 1/(ω0 sqrt(R_p R_t − R_t²)), R_p = Q_e ω0 L (탭은 낮추는 방향으로만 변환 - R_t ≥ R_p 는 계산기가 거부), 병렬 C_i = C_res − 결합 − 탭 등가 "
                "(Dishal 1949; Zverev 1967). 포트는 설계서의 BFR92 포트 모델 1 kΩ (콜렉터) / 500 Ω (다음 베이스)이지만, 콜렉터 노드에는 초크가 AC 접지로 병렬로 "
                "붙어 있고(1 µH 는 223.8 MHz 에서 약 1.4 kΩ - 포트와 같은 크기) 다음 베이스 노드에는 분압기가 붙어 있습니다. 그래서 탭은 "
                "Z = R_port ∥ (jω0 L_ch + ω0 L_ch / Q) 의 실수부 R' 를 변환하고 허수부 X' 를 흡수하며"
                "(C_s = 1/(ω0 (X' + sqrt(R' (R_p − R')))), `calc.rf.resonator.top_c.port_r` / `.port_x` / `.c_tap_reactive`), 부하 쪽은 500 Ω ∥ 분압기"
                "(`tx.tank<k>.r_load_eff`)로 설계합니다. 초크를 빼고 설계하면 x6 탱크의 저지가 한쪽에 2-3 dB 나빠집니다(ngspice-42). 초크와 분압기는 탱크 픽스처의 "
                "구성원이고 공칭값은 그 회로망의 정확한 응답(`calc.rf.resonator.top_c.ported_s21_db` / `.ported_rel_s21_db`)입니다.\n\n"
                "| 탱크 | f0 | L | BW | 원천 R' + jX' / 부하 | 탭 입/출 | 결합 | 병렬 1/2 | S21 (Q_u 40) | f0 ∓ f_T 상대 |\n|---|---|---|---|---|---|---|---|---|---|\n"
                + "\n".join(tank_rows) + "\n\n"
                "상단 결합 회로망은 비대칭이어서 + 쪽이 약합니다. 대칭 협대역 식은 + 쪽에서 최대 11.6 dB 어긋나므로 공칭값은 정확한 회로망 응답"
                "이고 허용오차 1 dB 입니다. 설계서는 S21 행에 Cohn 손실(4.34 dB)을 적었으나 정확한 회로망 값이 더 엄격한 공칭값이어서 그것을 쓰고 Cohn 값은 표에 "
                "함께 기록합니다(S21 에는 다음 단 분압기가 먹는 전력도 들어 있음). 탱크에는 트리머가 없습니다(BMI-S-105 차폐 안의 공간): 정렬은 부품 "
                "선별이며 실험 항목입니다. 체배기의 고조파 발생(C급)과 PM 재성장은 모델링하지 않습니다."
            )),
            TheorySection("출력 대역통과 필터, 패드, 드라이버", (
                f"×12 단의 콜렉터에서 드라이버 패드까지 하나의 {number(p('tx.bpf.n'))}공진기 상단 결합 필터: f_c, BW {q('tx.bpf.bw', 'Hz')}, L {q('tx.bpf.l', 'H')}, "
                f"원천 = 콜렉터 포트 ∥ 초크 (R' {q('tx_bpf.port_r', 'ohm')} + jX' {q('tx_bpf.port_x', 'ohm')}), 부하 = 드라이버 패드의 정합 입력 "
                f"{q('tx.z_mid', 'ohm')}. S21(f_c) = {number(p('tx_bpf.s21'), 4)} dB (Cohn {number(_neg(p('tx_bpf.loss')), 4)} dB); f_c ∓ f_T 에서 상대 "
                f"{number(p('tx_bpf.rel_m1'), 4)} / {number(p('tx_bpf.rel_p1'), 4)} dB - 판정은 정확한 값에서 1 dB 뺀 한계(설계서 규칙), f_c/2 에서 ≤ −50 dB, 2 f_c 에서 "
                "≤ −20 dB, ∓2 f_T · ∓3 f_T · 3/2 f_c 는 기록만 합니다.\n\n"
                "설계서는 마지막 이중 동조 탱크와 3단 대역통과 필터를 50 Ω 중간 임피던스로 이어 두 픽스처의 저지를 더했지만, 실제 회로에서 두 필터는 탭 커패시터 둘로 "
                "직접 이어지고 그 사이에 저항성 50 Ω 노드가 없습니다. 그렇게 이어진 회로망은 f_c + f_T 근처에 가짜 응답을 만들어 + 쪽 저지가 두 픽스처의 합보다 "
                "약 13 dB 나빴습니다(ngspice-42). 그래서 둘을 하나의 회로망으로 설계했고, 그 픽스처가 콜렉터에서 패드까지 실제 회로 그대로를 판정합니다.\n\n"
                f"정합된 π 패드 (`calc.rf.attenuator.pi.*`): 드라이버 입력 {q('tx.drv_pad.a_db', 'dB')} (R_shunt {q('drv_pad.r_shunt', 'ohm')}, R_series "
                f"{q('drv_pad.r_series', 'ohm')}), PA 구동 {q('tx.pa_pad.a_db', 'dB')} (R_shunt {q('pa_pad.r_shunt', 'ohm')}, R_series {q('pa_pad.r_series', 'ohm')}). "
                "PA 구동 패드는 전도 측정 뒤 다시 고르며 그것은 사람의 설계 변경입니다. PHA-1 은 TX_5V 에서 초크로 RF_OUT 에 바이어스하고 입출력에 DC 차단을 둡니다"
                "(이득과 출력 레벨은 검증되지 않음)."
            )),
            TheorySection("전력 증폭기, 부하선 정합, 고조파 저역통과 필터", (
                f"부하선: R_L = (V_CC − V_sat)² / (2P) = ({q('power.tx_5v', 'V')} − {q('pa.v_sat', 'V')})² / (2 · {q('pa.p_out', 'W')}) = {q('pa.r_l', 'ohm')} "
                "(`calc.rf.pa.load_line_r`, Cripps). 저역통과형 L 정합으로 R_L → 시스템 임피던스: "
                f"Q = sqrt(Z0/R_L − 1) = {number(q_m, 4)}, 직렬 L = {q('pa.match.l', 'H')} (PA 쪽), 병렬 C = {q('pa.match.c', 'F')} (50 Ω 쪽) "
                "(`calc.rf.lmatch.lowpass.*`), 그 뒤 DC 차단. 픽스처(출력 초크 포함): S21 ≥ −0.5 dB, S11 ≤ −15 dB.\n\n"
                f"고조파 저역통과: 7차 0.1 dB 체비셰프, 리플 모서리 {q('lpf.f_edge', 'Hz')}, 병렬 C 먼저, g = {g_rows} (`calc.rf.lpf.chebyshev.g`), 소자 {lpf_rows} "
                f"(`calc.rf.lpf.shunt_c` / `.series_l`). 무손실 감쇠 (`calc.rf.lpf.chebyshev.attenuation`): 2 f_c 에서 {number(att2, 4)} dB, 3 f_c 에서 "
                f"{number(att3, 4)} dB. 픽스처(Q_u 40): f_c 에서 S21 ≥ −1.5 dB, S11 ≤ −15 dB, 2 f_c 에서 ≤ −45 dB, 3 f_c 에서 ≤ −60 dB. 저지대역 판정은 각 병렬 "
                "소자의 패드에 접지면 비아가 있는 레이아웃에서만 성립합니다(447 MHz 에서 트랙 1 mm ≈ 1 nH).\n\n"
                "기판은 정합의 DC 차단과 저역통과 필터의 첫 병렬 C 를 LPF_IN 에서 저항성 노드 없이 바로 잇습니다. 두 픽스처는 각각 그곳을 시스템 임피던스로 "
                "가정했으므로, 캐스케이드 픽스처 `pa_lpf` 가 두 회로망의 부품 전체를 부하선(R_L)에서 출력까지 한 번에 판정합니다: f_c 에서 S21 ≥ "
                f"{q('pa_lpf.s21_min', 'dB')} (두 한계의 합, `calc.rf.db_sum`), 부하선에서 S11 ≤ {q('pa.match.s11_max', 'dB')}, 2 f_c / 3 f_c 에서 ≤ "
                f"{q('lpf.h2_max', 'dB')} / {q('lpf.h3_max', 'dB')}.\n\n"
                "PA(MMZ09332BT1)는 모델이 없어 설계 덱에서 공급 전류만 모사합니다(`model.pa.supply`): POWER_DOWN 이 문턱(`model.pa.v_pd`) 아래이면 VCC1 → GND "
                "저항 `model.pa.r_supply`, 위이면 `model.pa.r_off` 만. 그래서 P9 블록과 조합한 기판에서 PTT 해제 때 PA_5V 를 비우는 것은 PA 가 아니라 PTT 블록의 능동 방전(Q205)입니다(벤치 기판에서는 벤치 전원이 "
                "PA_5V 를 공급). "
                "447 MHz 적용 범위, 5 V 에서 0.5 W, 이득, POWER_DOWN 극성과 문턱, 바이어스 회로(VBA/VBIAS 저항)는 모두 검증되지 않았습니다."
            )),
            TheorySection("모델 값과 검증의 한계", (
                "이 기판의 PASS 는 **확인된 모델 값 아래의 회로망·원리 판정**입니다(실제 부품의 측정이 아님): 인덕터 Q (`model.l_q.*`), 바랙터 C(V) "
                "(`model.varactor.*`), 포트 저항(`model.tcxo.r_out`, `model.buf.*`, `model.bfr92.*`, `model.driver.port_r`, `model.pa.r_in`), 범용 NPN 카드"
                "(`model.npn`), PA 공급 전류 모델(`model.pa.r_supply` / `.r_off` / `.v_pd`). `rf.model_grounding` 은 이 값들이 근거를 얻을 때까지 NOT_VERIFIED 입니다.\n\n"
                "실험 항목(`rf.lab.*`): 반송파 주파수와 드리프트, TCXO 조달, 위상 변조기 선형성, 편이, 점유 대역폭, 체배 스퓨리어스, 탱크 정렬, PA 구동 레벨, "
                "출력 전력, 고조파, PA 안정성, 발열."
            )),
        ]

    def theory_figures(self, ir: CircuitIR) -> list[Figure]:
        """(a) a PM tank's exact phase against the varactor bias (the three states marked); (b) the three multiplier tanks' exact |S21| around their centres; (c) the band-pass and the lossless low-pass."""
        p = lambda k: self._p(ir, k)  # noqa: E731
        out: list[Figure] = []
        keys = ("tx.f_ref", "pm.l", "model.l_q.hf", "pm.c_fixed", "pm.c_trim", "model.tcxo.r_out", "pm.c_dc", "pm.r_s1", "pm.c_bypass", "pm.r_feed", "model.buf.r_in",
                "model.varactor.cjo", "model.varactor.vj", "model.varactor.m", "pm.v_lo", "pm.v_bias", "pm.v_hi")
        vals = [p(k) for k in keys]
        if None not in vals:
            f_t, l, q_u, c_fix, c_trim, r_port, c_dc, r_s, c_byp, r_feed, r_load, cjo, vj, m, v_lo, v0, v_hi = vals  # type: ignore[misc]

            def phase(v: float) -> float:
                return radio.pm_tank_phase_loaded_deg(f_t, f_t, l, q_u, c_fix, c_trim, radio.varactor_c_f(cjo, vj, m, v), r_port, c_dc, r_s, c_byp, r_feed, r_load)

            try:
                xs = _lin_grid(max(0.2, v0 - 1.0), v0 + 1.0, 201)
                ys = [phase(x) for x in xs]
                marks = [(v, phase(v), label) for v, label in ((v_lo, "bias_lo"), (v0, "bias_nom"), (v_hi, "bias_hi"))]
            except ValueError:
                xs, ys, marks = [], [], []
            if xs:
                caption = (f"'위상 변조기' 절의 탱크 1 의 정확한 위상 (`calc.rf.pm.tank_phase_loaded`, f_T = {quantity(f_t, 'Hz')}, 바랙터 C(V) 는 "
                           f"`calc.rf.varactor.c_at_bias`). 점 = 픽스처 상태 bias_lo / bias_nom / bias_hi. {THEORY_CURVE_NOTE}")
                out.append(_curve_figure("theory_pm_phase", "위상 변조 탱크 위상 대 바이어스", caption, [("탱크 위상", xs, ys)], x_label="바랙터 역바이어스 (V)",
                                         y_label="위상 (°)", markers=marks))
        series = []
        for k, stage in ((1, "x3"), (2, "x6")):
            ks = (f"tx.f{k}", f"tx.tank{k}.bw", f"tx.tank{k}.l", "model.bfr92.r_out", f"tx.{stage}.l_choke", "model.bfr92.r_in", f"tx.tank{k}.r_load_eff")
            v = [p(x) for x in ks]
            f0 = v[0]
            if None in v or f0 is None:
                continue
            try:
                q_u = p("model.l_q.vhf")
                if q_u is None:
                    continue
                xs = _lin_grid(0.5, 1.5, 201)
                ys = [radio.top_c_ported_s21_db_value(2, f0, v[1], v[2], v[3], v[4], q_u, v[5], v[6], q_u, x * f0) for x in xs]  # type: ignore[arg-type]
                series.append((f"tx_tank{k} ({quantity(f0, 'Hz')})", xs, ys))
            except ValueError:
                continue
        f_t, f1 = p("tx.f_ref"), p("tx.f1")
        if series and f_t and f1:
            caption = ("'체배기와 이중 동조 탱크' 절의 두 탱크의 정확한 S21 (`calc.rf.resonator.top_c.ported_s21_db`, Q_u 모델값, 콜렉터 초크와 다음 단 분압기 포함). "
                       "가로축은 각 탱크의 중심 주파수로 "
                       f"정규화한 f/f0; 안내선 = 탱크 1 의 f0 ∓ f_T. {THEORY_CURVE_NOTE}")
            out.append(_curve_figure("theory_tx_tanks", "체배 탱크 S21", caption, series, x_label="f / f0", y_label="S21 (dB)",
                                     bands=[("x", 1 - f_t / f1, 1 - f_t / f1, "f0 − f_T"), ("x", 1 + f_t / f1, 1 + f_t / f1, "f0 + f_T")]))
        f_c, n_b, bw, lb, r_out, l_ch, z_mid, q_u, f_e, rip = (p(k) for k in ("rf.f_c", "tx.bpf.n", "tx.bpf.bw", "tx.bpf.l", "model.bfr92.r_out", "tx.x12.l_choke",
                                                                              "tx.z_mid", "model.l_q.uhf", "lpf.f_edge", "lpf.ripple"))
        if None not in (f_c, n_b, bw, lb, r_out, l_ch, z_mid, q_u, f_e, rip):
            try:
                xs = _lin_grid(150e6, 1500e6, 271)
                bpf = [max(-120.0, radio.top_c_ported_s21_db_value(n_b, f_c, bw, lb, r_out, l_ch, q_u, z_mid, z_mid, q_u, x)) for x in xs]  # type: ignore[arg-type]
                lpf = [max(-120.0, -chebyshev_attenuation_db(7, rip, x, f_e)) for x in xs]  # type: ignore[arg-type]
            except ValueError:
                xs = []
            if xs:
                caption = (f"'출력 대역통과 필터' 절의 콜렉터-패드 필터의 정확한 S21 (`calc.rf.resonator.top_c.ported_s21_db`, Q_u 모델값, 콜렉터 초크 포함) 과 "
                           f"'고조파 저역통과 필터' 절의 무손실 "
                           f"7차 체비셰프 감쇠 (`calc.rf.lpf.chebyshev.attenuation`); 안내선 = f_c, 2 f_c, 3 f_c. {THEORY_CURVE_NOTE}")
                out.append(_curve_figure("theory_tx_filters", "대역통과·저역통과 필터 응답", caption, [("tx_bpf", xs, bpf), ("lpf (무손실)", xs, lpf)],
                                         x_label="주파수 (Hz)", y_label="S21 (dB)",
                                         bands=[("x", f_c, f_c, "f_c"), ("x", 2 * f_c, 2 * f_c, "2 f_c"), ("x", 3 * f_c, 3 * f_c, "3 f_c")]))  # type: ignore[operator]
        return out

    def part_notes(self, ir: CircuitIR) -> dict[str, PartNote]:
        from ai_eda.design.rf.t_audio_ptt import stage1_part_notes, stage1_refs

        p = lambda k: self._p(ir, k)  # noqa: E731
        stage1 = stage1_refs(ir)  # the P9 companions' parts: the stage-1 board's notes (their descriptions would match the TX chain's patterns)
        out: dict[str, PartNote] = stage1_part_notes(ir, stage1)
        for c in ir.components:
            if c.ref in stage1:
                continue
            d = c.description
            lib = f"{c.symbol.library}:{c.symbol.name}" if c.symbol is not None else NO_RECORD
            fp = f"{c.footprint.library}:{c.footprint.name}" if c.footprint is not None else NO_RECORD
            if "shield can" in d:  # first: the modulator's can names the PM tanks it covers
                can = c.value if c.value.startswith("BMI-S-") else "BMI-S"
                what = "위상 변조기(TCXO, 탱크, 팔로워)를 가림" if "phase modulator" in d else "체배기 세 단을 가림"
                size = {"BMI-S-103": "26.21 × 26.21 mm", "BMI-S-105": "38.1 × 25.4 mm"}.get(can, fp)
                out[c.ref] = PartNote(f"차폐 캔 ({can})", f"{what} (펜스 패드는 GND)", [size, "일체형", "펜스 틈으로 DC·오디오 배선 통과"], [unverified(f"Laird {can}")])
            elif "TX reference TCXO" in d:
                out[c.ref] = PartNote("송신 기준 TCXO (f_c / 12)", f"KT2520K-T 풋프린트의 맞춤 주파수 {quantity(p('tx.f_ref'), 'Hz')}; 라이브러리 설명 '10-60MHz' 안",
                                      [f"주파수 {quantity(p('tx.f_ref'), 'Hz')}", f"안정도 ≤ {number(p('rf.frequency_tolerance') or p('rf.tcxo_stability'))} ppm", "3.3 V 공급, 클립 사인 출력",
                                       "채널을 원문으로 확인하기 전 주문 금지"], [unverified("Kyocera KT2520K 맞춤 주문"), unverified("Epson TG2520 계열", "핀 배열 확인")])
            elif "varactor" in d:
                out[c.ref] = PartNote("위상 변조 바랙터", "탱크 커패시턴스의 가변 부분; 모델은 `model.varactor.*` (검증되지 않음)",
                                      [f"V0 = {quantity(p('pm.v_bias'), 'V')} 에서 약 {quantity(p('pm.c_var.nom'), 'F')}", "37 MHz 에서 직렬저항 작음", "SOD-323"],
                                      [unverified("BB208 / SMV1233 계열", "C(V) 곡선을 측정해 model.varactor 를 교체")])
            elif "PM tank" in d:
                out[c.ref] = PartNote("위상 변조 탱크 소자", f"계산기 값 그대로(E 계열 반올림 없음): {c.value}",
                                      [f"값 {c.value}", "커패시터 NP0/C0G", "인덕터 Q ≥ 40 @ 37 MHz (모델값)"] + (["트리머 범위가 중간 위치를 포함"] if c.symbol is not None and c.symbol.name == "C_Trim" else []),
                                      [unverified("0402 NP0 / 0603 권선형 인덕터 / Murata TZB4 트리머")])
            elif "PM buffer" in d:
                out[c.ref] = PartNote("위상 변조 탱크 뒤의 버퍼(이미터 팔로워) 부품",
                                      "탱크의 부하를 `model.buf.r_in` 으로 고정 - 분압기와 이미터 저항은 `model.buf.r_in` / `.r_out` 포트 모델 안에 있음",
                                      [f"값 {c.value}"], [unverified("BC847 / 0402 저항")])
            elif "multiplier stage" in d:
                out[c.ref] = PartNote("체배기 단 부품 (BFR92 와 바이어스)", "분압 바이어스, 에미터 R // C, 0 Ω 링크(전류 측정), 콜렉터 초크 - 초크와 분압기는 탱크·필터 설계에 흡수됨",
                                      [f"값 {c.value}", "BFR92: f_T ≥ 5 GHz (라이브러리 설명)", "초크: 값과 Q 가 설계값에 맞을 것(탱크 탭이 그것을 흡수), 자기공진이 단 출력 주파수보다 높을 것"],
                                      [unverified("BFR93A / BFS17", "SOT-23 핀 배열 확인")])
            elif "tx_tank" in d or "tx_bpf" in d:
                out[c.ref] = PartNote("상단 결합 탱크 / 대역통과 필터 소자", "`calc.rf.resonator.top_c.*` 계산값 그대로",
                                      [f"값 {c.value}", "커패시터 NP0 ±0.05 pF 급 또는 선별", "인덕터 Q ≥ 40 (모델값), 0604HQ 급(UHF)"],
                                      [unverified("Coilcraft 0604HQ / 0603HP, Murata GJM 0402")])
            elif "pad" in d:
                out[c.ref] = PartNote("정합 π 패드 저항", "`calc.rf.attenuator.pi.*`", [f"값 {c.value}", "0402 박막 1 %", "PA 구동 패드는 전력 정격 확인"], [unverified("0402 1 % 박막")])
            elif "TX driver" in d:
                out[c.ref] = PartNote("송신 드라이버 (PHA-1)", "50-6000 MHz 이득 블록 (라이브러리 설명); SPICE 모델 없음", ["447 MHz 이득·P1dB 확인", "5 V 초크 바이어스"],
                                      [unverified("Mini-Circuits PSA4-5043+", "핀 배열 확인")])
            elif "power amplifier" in d:
                out[c.ref] = PartNote("전력 증폭기 (MMZ09332BT1)", "400-1000 MHz 급 2단 PA (검증되지 않음); SPICE 모델 없음, 설계 덱에서는 공급 전류 저항",
                                      ["447 MHz 에서 0.5 W 이상", "5 V 공급", "POWER_DOWN 극성 확인", "EP 방열 비아"],
                                      [unverified("CML CMX901", "라이브러리에 있음(pa_alt), 핀·바이어스 다름"), unverified("RF Micro RF5110G")])
            elif "load-line" in d or "output DC block" in d or "output choke" in d:
                out[c.ref] = PartNote("부하선 정합 / 출력 초크 / DC 차단", "`calc.rf.pa.load_line_r` 와 `calc.rf.lmatch.lowpass.*`",
                                      [f"값 {c.value}", "고주파 전류 정격 (0.5 W, 20 Ω 부하선에서 약 0.2 A)", "고 Q"], [unverified("Coilcraft 0604HQ / NP0 0402")])
            elif "harmonic low-pass" in d:
                out[c.ref] = PartNote("고조파 저역통과 필터 소자", "7차 0.1 dB 체비셰프 (`calc.rf.lpf.*`)", [f"값 {c.value}", "NP0, 고 Q 인덕터", "병렬 소자마다 접지면 비아"],
                                      [unverified("Mini-Circuits LFCN-490 (라이브러리에 있음, lpf_alt)", "삽입손실·저지대역 확인")])
            elif "conducted RF output" in d:
                out[c.ref] = PartNote("전도 출력 커넥터 (U.FL)", "감쇠기 / 더미 로드 전용 - 안테나 금지", ["50 Ω", "0.5 W 에서 U.FL 정격 확인"], [unverified("Hirose U.FL-R-SMT-1(10)")])
            elif "bench header" in d:
                out[c.ref] = PartNote("벤치 인터페이스 헤더 (P9 전원·PTT·오디오 블록 대체)", d, ["2.54 mm 핀 헤더"], [unverified("일반 1x3 핀 헤더")])
            elif c.symbol is not None and c.symbol.name in ("R", "C", "L", "C_Polarized", "Fuse"):
                kind = {"R": "저항", "C": "커패시터", "L": "인덕터", "C_Polarized": "전해 커패시터", "Fuse": "퓨즈"}[c.symbol.name]
                out[c.ref] = PartNote(f"{c.ref}: {d}", f"{kind} {c.value} ({lib}, {fp}); 계산기 출력 또는 확인된 선택값(E 계열 반올림 없음)",
                                      [f"값 {c.value}", "같은 풋프린트", "정격 전압·전류 확인"], [unverified(f"같은 값의 다른 제조사 {kind}")])
            else:
                out[c.ref] = PartNote(f"{c.ref}: {d}", f"{lib} / {fp} (라이브러리에서 핀 이름으로 배선).", ["핀 배열과 정격을 데이터시트로 확인"], [])
        return out


def _neg(x: float | None) -> float | None:
    return None if x is None else -x


KR447_TX_EXCITER = Kr447TxExciterTemplate()

__all__ = [
    "ANTENNA_KEYS",
    "BANDWIDTH_RANGE",
    "BUILD",
    "DEVIATION_RANGE",
    "KR447_TX_EXCITER",
    "PLANE_REASON",
    "PREFIXES",
    "RADIO_BUILD",
    "REGIONS",
    "TEMPLATE_ID",
    "TOT_RANGE_S",
    "V_IN_RANGE",
    "BenchTxBlock",
    "Companions",
    "Kr447TxExciterTemplate",
    "bench_companions",
    "default_companions",
    "tx_blocks",
    "tx_net_classes",
]
