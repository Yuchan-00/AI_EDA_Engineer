"""The ``kr447_transceiver`` template (``radio_build = transceiver`` / ``transceiver_conducted``): the stage-5 composition of the KR 447 MHz FM radio (kr447 design §2.5).

Invariant: a pure function of the confirmed requirements, the library on
disk and the block builders - nothing is guessed, nothing is grounded that is
not. The board is the composition (:func:`~ai_eda.design.rf.blocks.base.merge_results`)
of every block of the four stage boards, built by the same builders with the
same numbering (a PASS never carries over from a stage board: the composed IR
hashes differently and every check runs again on it):

* stage 1 (part P9): ``PowerBlock()`` (1xx, both modes), ``PttBlock(pa_stand_in=False)``
  (2xx; the 4060 time-out only with ``tx_timeout``), ``TxAudioBlock(k_pm_key="tx.k_pm")``
  (3xx, built last: it designs its integrator with the modulator's slope),
  ``RxAudioBlock()`` (4xx) - composed without net prefixes, exactly as on the
  stage-1 board;
* stage 2 (P10): ``IfBackendBlock(input_connector=False)`` (5xx, internal nets
  ``IFB_*``: its ``LIM_OUT`` / ``MIX_IN`` names are the TX audio's and the
  mixer's too) - the front end drives ``IF1``;
* stage 3 (P11): ``RxFrontendBlock()`` / ``RxMixerBlock()`` (6xx) and
  ``LoChainBlock()`` / ``LoBufferBlock()`` (7xx) - the T/R switch drives
  ``RX_RF``;
* stage 4 (P12): ``TxModBlock()`` / ``TxChainBlock()`` / ``TxDriverBlock()``
  (8xx) and ``PaBlock(with_lpf=False, with_output=False, out_net="TX_RF")``
  (9xx: the low-pass moves behind the switch, the U.FL goes);
* stage 5 (P13, :mod:`ai_eda.design.rf.blocks.trx`): ``TrxBlock`` (10xx: the
  PIN T/R switch, the harmonic low-pass, the antenna L-match when a confirmed
  ``antenna_impedance`` differs from the system impedance) and
  ``AntennaBlock`` - ``ANT1``, the integral straight quarter-wave wire
  (``transceiver``), or ``J1001``, a U.FL in its place (``transceiver_conducted``:
  bench and KC conducted samples, never fitted with an antenna).

The two builds differ exactly in the part at the antenna feed (``ANT1`` /
``J1001``) and in what goes with it: the antenna's length row
(``trx.ant.vf`` / ``trx.ant.length``), its lab items (length, radiation,
radiated spurious, SAR) against the conducted sample's, the antenna match
(only on ``transceiver``: the conducted sample's U.FL is the system-impedance
port the instruments expect), the antenna band (its keep-out and its height
row ``floor.ant_band.h``: the band keeps the radiator's base clear, and the
conducted sample has no radiator - its U.FL's ground pads also need their
plane vias there), the antenna block's title and one constraint; every other
part, net, choice, region and fixture is the same.

Selection and the closed world (kr447 design §2.0): selected only by the
confirmed categorical requirement ``radio_build`` = ``transceiver`` or
``transceiver_conducted`` (:meth:`triggered_by`); it needs
``carrier_frequency`` (a channel of the unverified KR raster, else a refusal
naming the channel list), ``modulation`` (``fm``) and ``input_voltage``
(6.6-8.4 V: above the low-pack TX inhibit's release point, at most a full 2S
pack); it serves every key of ``family.BUILDS["transceiver"]``
(``tx_timeout``, ``erp`` / ``eirp``, the antenna keys, ``rx_sensitivity`` and
``system_impedance`` optional) and refuses every other confirmed design
requirement with the family's sentence. A stated ``pcb_layers`` other than
4 refuses before anything is asked (the layer policy: the RF lines need a
reference plane). A stated ``tx_power`` above the profile's
``kr447.max_power`` placeholder builds and ``rf.regulatory_profile`` then
FAILs; one outside 0.01-1 W, a deviation / audio bandwidth / time-out
outside the stage boards' design ranges and an ``antenna_impedance`` outside
10-250 ohm are refused.

What the plan carries besides the blocks' own content: every ``kr447.*``
regulatory placeholder as a choice whose description ends
``[UNVERIFIED: ...]`` (``ir.rf.profile_keys``), the floorplan regions
(``floor.<block>.x`` / ``.y`` / ``.w`` / ``.h`` mm, choices), the antenna
band's height ``floor.ant_band.h``, the single-tone 99 % occupied bandwidth
at 1 kHz / 3 kHz (``calc.rf.fm.obw99``, numbers without a verdict), the
receive sensitivity estimated at the antenna port (``trx.nf.total`` /
``trx.sensitivity``: the front end's Friis budget plus the confirmed choice
``trx.rx.front_loss`` for the low-pass and the switch - an estimate, never a
verdict; decision 2A's about -113 dBm), the radiated numbers when their
inputs are stated (``ant.eirp_dbm`` / ``ant.erp_dbm`` from ``antenna_gain``,
``ant.fspl`` / ``ant.e_at_range`` with ``link_range``: numbers, the verdict
is the lab's), the frequency plan's gated row ``tx_ref_on_rx_channel`` (N x
f_T = f_c exactly: the TX reference and its chain must be unpowered in RX -
it points to ``spice.tx_rail_off_rx`` and the lab item ``tr_sequencing``),
the keep-outs (:attr:`~ai_eda.design.base.Plan.keepouts`, written into
``ir.pcb.keepouts`` by :func:`~ai_eda.design.board.add_board`) and the
signal-integrity classes of every stage (:meth:`si_declarations`).

Board (kr447 design §2.5, as updated; the numbers are the floorplan choices
of :data:`REGIONS`): 60 mm wide like the design, the antenna end at the top.
The design's 60 x 145 mm outline cannot hold the composed board (a strict
bound, ``tests/test_kr447_transceiver.py``): the extents of the parts outside
the cans and of the cans, each grown by the placer's 1 mm spacing, sum to
about 8.5e3 mm^2, while the outline leaves 57 x 142 = 8.1e3 mm^2 inside its
2 mm edge margin (grown by the same spacing; the antenna band not even
counted); and the stage blocks' cans are four (the front end and the modulator each got
their own BMI-S-103 in parts P11 / P12) where the design drew three, one of
them a BMI-S-102. Two BMI-S-103 cans side by side need 60.04 mm of width
with the placer's margins, so every can has its own row and the outline is
60 x 245 mm (measured: ``placement.rf_floorplan`` places both builds' 441
parts, 453 with the time-out and the antenna match, in about 0.1 s; the PTT
region holds the 4060 time-out and the PA_5V discharge on every build; the
PA and TX audio regions hold the fan-out room of their fine-pitch parts).
Edge placement is a deviation from kr447 design §2.5 ("VOL / SQL pots and
the mic on the bottom edge", with the pack and speaker connectors): the
placer packs each block's parts from its region's top-left corner, so on
both builds (measured on the KiCad 10.0.6 footprints, 2 mm edge margin)
``SW201`` (PTT) is on the left edge as §2.5 asks, but ``RV401`` (volume) is
44.2 mm and ``RV402`` (squelch) 29.9 mm above the bottom edge (the RX audio
region), ``MK301`` is on the left edge 171 mm above it (the TX audio
region beside the LO chain), ``J401`` (speaker) is 52 mm above it and 8.2 mm
from the right edge, and ``J101`` (pack) is on the left edge 19 mm above it.
Moving them is left to manual placement in KiCad (an edge region of their
own would take the parts out of their blocks);
``tests/test_kr447_transceiver.py`` pins these positions. Keep-outs: ``ant_band``
(``transceiver`` only) - the top ``floor.ant_band.h`` mm, every copper
layer, no tracks / vias / pads / footprints except ``ANT1`` and the
``ANT_FEED`` net; the planes stay allowed up to their edge inset, so the
monopole's counterpoise reaches the feed (layout (a) of the design's critic2
note: the feed pad sits at the top edge where the plane ends, within the
placer's 2 mm margin - the wire's first millimetres over the plane are part
of the lab item ``antenna_length``); ``trx_bcu`` - no B.Cu tracks under the
trx region (the quarter-wave and low-pass inductors; the packer decides
where in the region they sit, so the whole region is kept) except GND:
``routing.maze`` folds a track ban into its cell map, which also bars a
via's B.Cu pad, and without the exemption no shunt part deeper than the
router's via reach inside the region gets the ground via
``c.kr447.rf_ground_vias`` asks for (measured: ``C1001``'s GND pad on the
reduced antenna-end board). A GND track on B.Cu would be ground under
ground (both inner layers are GND planes); the plane net gets pad vias,
never tracks.

Expected statuses on this machine (ngspice-42, the packed KiCad 10.0.6
libraries, no kicad-cli; ``tests/test_kr447_transceiver.py``): every
``spice.rf.*`` row of the 23 fixture networks (24 with the antenna match)
PASS as "a network verdict under confirmed model values (not a measured
part)" (the trx block's ``trsw`` / ``lpf`` among them, and the
composition-level cascade ``ant_end`` - the match, the switch, the low-pass
and ``fe_bpf2`` as the board joins them, :func:`~ai_eda.design.rf.blocks.trx.ant_end_result`,
the fixture-membership pass, part A1); the design deck's rows (the stage-1 rows,
the RF transistors' bias, ``pm_couple_*`` and ``pin_bias``) PASS;
``block.interface.*`` PASS (IR arithmetic); ``rf.freq_plan`` NOT_VERIFIED
(its margin rows PASS; the response / gated rows point to lab items);
``rf.model_grounding`` / ``rf.regulatory_profile`` / ``rf.lab.*``
NOT_VERIFIED; RELEASE NOT_VERIFIED at best. The board places and its router
result is whole (decision 2A): the floorplan leaves the MAX9814 (``U301``,
DFN-14, 0.4 mm pitch) and the PA (``U901``, QFN-12, 0.5 mm) their fan-out
room (``placement.rf_floorplan`` 0.2: every other part 2.2 mm from their
pads), ``routing.maze`` reads the two PHA-1s' ``SOT-89-3`` custom pads (the
LO buffer's ``U750``, the TX driver's ``U850``) as the boxes of their anchor
and primitives and joins every U301 / U901 pad whose own grid cell does not
serve to its 0.2 mm grid by an escape stub (0.6); 0.6's first pass still
left U301 pads without one (``U302.2``'s ground via walked into U301's
escape area), so the static phase ran again as routing.maze 0.7 - U301's
fan-out room kept clear of the other footprints' plane vias (``U302.2`` and
``SH801.1`` took other sites), its escapes' ways out running to the room's
edge - and every pad escapes; with 211 nets to route the board is a large
one and negotiates up to 150 iterations. Measured 2026-09-30 on both
builds' placed boards (the router alone, four routes sharing 4 CPUs): 211
of 211 nets, legal after 123 iterations (the antenna build 4428 tracks,
792 vias, 10064.2 mm, about 1080 s; the conducted one 4424 / 798 /
10090.6 mm, about 630 s); at the old 40-iteration cap 5 / 6 conflicting
nets were left. Through the whole pipeline (2026-09-30) five nets
(PTT_ACTIVE, PA_PD, MUTE, SQ_SET, PM_DRIVE) are promoted to Z50 and the
single re-route is legal after 98 / 104 iterations and kept (4259 / 4270
tracks, 770 / 776 vias), ``si.impedance.Z50`` PASSes, no check FAILs and
RELEASE is NOT_VERIFIED. The microphone ``MK301``
is the through-hole ``Sensor_Audio:POM-2244P-C3310-2-R`` (decision 1A;
the symbol's pin 1 "-" / pin 2 "+" land on pads 1 / 2, and which terminal
is the case is UNVERIFIED against the PUI Audio datasheet): the SMT ``CUI_CMC-4013-SMT`` it replaced has its pad 2 inside the ring of
its custom pad 1, which left ``MIC_P`` no way out on F.Cu without a via in
the pad (wave 2c's route of this board with it: 206 of 211 nets). The same
blocks route on their own (the antenna end, ``tests/test_kr447_transceiver.py``).
``pcb.keepout`` on the placed board judges every keep-out, region and can
fence and PASSes (the two custom pads bounded by the boxes of their parts).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ai_eda.ir import Block as TopologyBlock
from ai_eda.ir import CircuitDomain, CircuitIR, Constraint, ConstraintKind, Keepout, MissingInformation, NetClass, SimulationSetup, Topology, Traced
from ai_eda.ir.rf import PlanLine, RFBlock, RFDesign, RFRegion
from ai_eda.tools.calc import radio
from ai_eda.tools.calc.rf import dbm_to_w, eirp, eirp_to_erp, field_strength, fspl, sensitivity, w_to_dbm
from ai_eda.tools.kicad.library import KicadLibrary

from ai_eda.design.base import (
    DesignChange,
    LayerPolicy,
    PartNote,
    Plan,
    Template,
    TheorySection,
    choice_provenance,
    number,
    parameter_value,
    quantity,
    requirement_text,
    unserved_requirements,
    unverified,
)
from ai_eda.design.inputs import DesignInput, canonical_key, present_keys, read_modulation, read_radio_build
from ai_eda.design.library_parts import TemplateRefusal
from ai_eda.design.rf.blocks.base import GROUND_NET, Block, BlockBuilder, BlockContext, BlockPrefix, BlockResult, exclude_floating, merge_results
from ai_eda.design.rf.blocks.if_backend import BLOCK_ID as IF_BLOCK_ID
from ai_eda.design.rf.blocks.if_backend import IfBackendBlock
from ai_eda.design.rf.blocks.lo_chain import BUFFER_ID as LO_BUFFER_ID
from ai_eda.design.rf.blocks.lo_chain import CHAIN_ID as LO_CHAIN_ID
from ai_eda.design.rf.blocks.lo_chain import LoBufferBlock, LoChainBlock
from ai_eda.design.rf.blocks.pa import P_OUT_RANGE_W, PA_ID, PaBlock
from ai_eda.design.rf.blocks.power import PACK_CUTOFF_V, PACK_MAX_V, PowerBlock
from ai_eda.design.rf.blocks.ptt import PttBlock
from ai_eda.design.rf.blocks.rx_audio import RxAudioBlock
from ai_eda.design.rf.blocks.rx_frontend import FRONTEND_ID, MIXER_ID, RxFrontendBlock, RxMixerBlock
from ai_eda.design.rf.blocks.trx import (
    ANTENNA_ID,
    ANT_R_RANGE,
    FEED_NET,
    FEED_REF,
    TRX_ID,
    TX_NET,
    VARIANT_ANTENNA,
    VARIANT_CONDUCTED,
    AntennaBlock,
    TrxBlock,
    ant_end_result,
)
from ai_eda.design.rf.blocks.tx_audio import TxAudioBlock
from ai_eda.design.rf.blocks.tx_chain import CHAIN_ID as TX_CHAIN_ID
from ai_eda.design.rf.blocks.tx_chain import DRIVER_ID, K_PM_KEY, MOD_ID, TxChainBlock, TxDriverBlock, TxModBlock, calc, copy_input
from ai_eda.design.rf.family import BUILDS, unserved_message
from ai_eda.design.rf.parts import part_line
from ai_eda.design.rf.profile import PROFILE, PROFILE_BY_KEY, profile_choices, profile_keys, raster_refusal
from ai_eda.design.templates import THEORY_CURVE_NOTE, _changes, _curve_figure, _lin_grid, _missing_inputs, _net_line, _refused

if TYPE_CHECKING:
    from ai_eda.design.board import BoardContext, SIDeclarations
    from ai_eda.report.figures import Figure

#: the two builds this template makes (the family's ``transceiver`` and its conducted variant)
RADIO_BUILD = VARIANT_ANTENNA
VARIANTS: tuple[str, str] = (VARIANT_ANTENNA, VARIANT_CONDUCTED)
BUILD = BUILDS[RADIO_BUILD]
TEMPLATE_ID = BUILD.template_id
#: why only 4 layers (kr447 design §2.0, decision 7A)
PLANE_REASON = "the RF lines need a reference plane (decision 7A: 4 layers)"
#: the pack voltages the board is designed for: above the low-pack TX inhibit's release point (cut-off + 0.2 V hysteresis), at most a full 2S pack
V_IN_RANGE: tuple[float, float] = (PACK_CUTOFF_V + 0.2, PACK_MAX_V)
#: the deviation / audio-bandwidth / time-out ranges of the stage-1 audio and PTT blocks (Hz, Hz, s)
DEVIATION_RANGE: tuple[float, float] = (500.0, 5000.0)
BANDWIDTH_RANGE: tuple[float, float] = (2000.0, 4000.0)
TOT_RANGE_S: tuple[float, float] = (10.0, 600.0)
TX_TIMEOUT_KEY = "tx_timeout"
#: the parameter the TX audio block holds the confirmed frequency_deviation in; the occupied-bandwidth rows read the same key
DEVIATION_PARAM = "tx.frequency_deviation"
#: how each block is numbered: references re-based by hundreds as the design numbers them; only the IF back-end's internal nets are prefixed
PREFIXES: dict[str, BlockPrefix] = {
    "power": BlockPrefix(100), "ptt": BlockPrefix(200), "tx_audio": BlockPrefix(300), "rx_audio": BlockPrefix(400),
    IF_BLOCK_ID: BlockPrefix(500, "IFB_"), FRONTEND_ID: BlockPrefix(600), MIXER_ID: BlockPrefix(600),
    LO_CHAIN_ID: BlockPrefix(700), LO_BUFFER_ID: BlockPrefix(700), MOD_ID: BlockPrefix(800), TX_CHAIN_ID: BlockPrefix(800),
    DRIVER_ID: BlockPrefix(800), PA_ID: BlockPrefix(900), TRX_ID: BlockPrefix(1000), ANTENNA_ID: BlockPrefix(0),
}
#: the board width (the design's 60 mm) and the antenna band's height (the critic's 6 mm)
BOARD_W_MM = 60.0
BAND_H_MM = 6.0
#: the floorplan region of each block (x, y, w, h mm from the board's top-left corner, Y down): choices ``floor.<block>.*``; the board is their
#: bounding box, 60 x 245 mm. Rows from the antenna end: the feed in the band; the switch / low-pass, the PA and the driver beside the front
#: end's can; the multipliers' BMI-S-105 beside the mixer and the LO buffer; the TX audio beside the LO chain's can; the PTT block beside the
#: modulator's can; then the IF back-end, the RX audio and the power section across the width. A BMI-S-103 region is 30.5 mm wide at the
#: right edge (27.52 mm can + the 2 mm edge margin + 0.5 mm spacing, 0.49 mm to spare); every region packs both builds (measured on the
#: KiCad 10.0.6 footprints). The fan-out room ``placement.rf_floorplan`` 0.2 keeps around the fine-pitch parts (2.2 mm from their pads at
#: the fine rules) sized two regions: the PA's (QFN-12 U901) is 11 mm tall, taken from the driver's (9 mm), and the TX audio region
#: (DFN-14 U301, and the through-hole microphone's 6.5 x 6.5 mm courtyard) 33 mm (3 mm more), the PTT region below it 37 mm (1 mm less
#: than before; it still holds the time-out and the PA_5V discharge), so every row below them is 2 mm lower.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    ANTENNA_ID: (6.0, 0.0, 8.0, 8.0),
    TRX_ID: (0.0, 8.0, 29.5, 10.0),
    PA_ID: (0.0, 18.0, 29.5, 11.0),
    DRIVER_ID: (0.0, 29.0, 29.5, 9.0),
    FRONTEND_ID: (29.5, 6.0, 30.5, 32.0),
    TX_CHAIN_ID: (0.0, 38.0, 42.0, 29.0),
    MIXER_ID: (42.0, 38.0, 18.0, 18.0),
    LO_BUFFER_ID: (42.0, 56.0, 18.0, 11.0),
    "tx_audio": (0.0, 67.0, 29.5, 33.0),
    LO_CHAIN_ID: (29.5, 67.0, 30.5, 30.0),
    "ptt": (0.0, 100.0, 29.5, 37.0),
    MOD_ID: (29.5, 97.0, 30.5, 30.0),
    IF_BLOCK_ID: (0.0, 137.0, 60.0, 50.0),
    "rx_audio": (0.0, 187.0, 60.0, 33.0),
    "power": (0.0, 220.0, 60.0, 25.0),
}
_REGION_WHAT = {
    ANTENNA_ID: "the part at the antenna feed, at the top edge in the antenna band",
    TRX_ID: "the PIN T/R switch, the harmonic low-pass and the antenna match below the feed",
    PA_ID: "the PA and its load-line match, next to the T/R switch",
    DRIVER_ID: "the pads and the PHA-1 driver above the multipliers",
    FRONTEND_ID: "the RX front end under its BMI-S-103 can, next to the T/R switch's RX side",
    TX_CHAIN_ID: "the x3 / x2 / x2 multipliers and the TX output band-pass under the BMI-S-105 can",
    MIXER_ID: "the ADEX-10 mixer, the diplexer and the IF1 post-amp below the front end",
    LO_BUFFER_ID: "the PHA-1 buffer and the pad into the mixer's LO port",
    "tx_audio": "TX audio (microphone, limiter, splatter filter, integrator) beside the LO chain (the microphone lands on the left edge here, not the bottom edge)",
    LO_CHAIN_ID: "the LO1 chain and its output band-pass under its BMI-S-103 can",
    "ptt": "PTT sequencer, TX interlock and PA supply switch beside the modulator",
    MOD_ID: "the TCXO and the two PM tanks under their BMI-S-103 can",
    IF_BLOCK_ID: "the IF back-end (crystal ladder, SA605D, IF2 filters, quadrature) across the width",
    "rx_audio": "RX audio, volume and squelch across the width (the pots and the speaker jack land 30-52 mm above the bottom edge, not on it)",
    "power": "pack connector, main switch, rail switches and regulators along the bottom edge (the pack connector lands on the left edge, 19 mm above the bottom)",
}
#: the requirement keys only the stage-1 audio / PTT blocks read (their design ranges) and the power / PA ranges
_RANGES: tuple[tuple[str, tuple[float, float], str], ...] = (
    ("input_voltage", V_IN_RANGE, "the 2S Li-ion pack above the confirmed cut-off 6.4 V plus the low-pack inhibit's 0.2 V hysteresis, at most a full pack"),
    ("frequency_deviation", DEVIATION_RANGE, "the limiter / integrator chain's design range (the KR profile's maximum is a separate check)"),
    ("audio_bandwidth", BANDWIDTH_RANGE, "the splatter and RX low-pass filters' design range (a voice channel)"),
    (TX_TIMEOUT_KEY, TOT_RANGE_S, "the 4060 RC time-out's design range"),
    ("tx_power", P_OUT_RANGE_W, "the MMZ09332BT1 stage's class [UNVERIFIED: NXP MMZ09332B datasheet] (the profile's limit is a separate check)"),
    ("antenna_impedance", ANT_R_RANGE, "the range an L-match of loaded Q at most 2 covers from the system impedance"),
)
#: stage-board theory sections this board does not reuse: the bench boards' overviews and model limits (this board's own sections say both) and
#: the stage-1 board's deviation section, whose K_pm is the model constant model.k_pm (here the pm_mod fixtures measure it: the stage-4 section)
_SKIPPED_STAGE_SECTIONS: tuple[str, ...] = ("개요", "모델 값과 검증의 한계", "적분기와 주파수 편이")
#: stage-board sentences that describe the bench board's stand-in and read differently on the composed radio (the stage-1 board has no PA)
_STAGE_TEXT_ON_THIS_BOARD: tuple[tuple[str, str], ...] = (
    ("(`pa_supply_off_first`, 이 기판에는 PA 가 없어 R219 가 부하를 대신함)",
     "(`pa_supply_off_first`, 이 기판에서는 해제와 같은 순간 PA 가 POWER_DOWN 으로 꺼져 전류를 거의 쓰지 않으므로(`model.pa.supply`), "
     "PA_PD 로 켜지는 능동 방전 Q205 / R224 가 PA_5V 를 비움)"),
)
#: blocks whose parts are the RF signal path (the topology's RF domain)
RF_BLOCK_IDS: frozenset[str] = frozenset({IF_BLOCK_ID, FRONTEND_ID, MIXER_ID, LO_CHAIN_ID, LO_BUFFER_ID, MOD_ID, TX_CHAIN_ID, DRIVER_ID, PA_ID, TRX_ID, ANTENNA_ID})


def transceiver_blocks(variant: str, *, tot: bool, with_match: bool) -> list[tuple[Block, BlockPrefix]]:
    """Every block of the board with its prefix, in build order (the supply first, the LO chain before the front end, the front end before the
    mixer's budget, the modulator before the TX audio's integrator)."""
    blocks: list[Block] = [
        PowerBlock(), PttBlock(pa_stand_in=False, tot=tot), LoChainBlock(), LoBufferBlock(), RxFrontendBlock(), RxMixerBlock(),
        IfBackendBlock(input_connector=False), RxAudioBlock(), TxModBlock(), TxChainBlock(), TxDriverBlock(),
        PaBlock(with_lpf=False, with_output=False, out_net=TX_NET), TrxBlock(with_match=with_match), AntennaBlock(variant), TxAudioBlock(k_pm_key=K_PM_KEY),
    ]
    return [(blk, PREFIXES[blk.id]) for blk in blocks]


def _p(ir: CircuitIR, key: str) -> float | None:
    return parameter_value(ir, key)


class Kr447TransceiverTemplate(Template):
    """``radio_build = transceiver`` / ``transceiver_conducted``: the whole radio - every stage block plus the T/R switch and the antenna end."""

    id = TEMPLATE_ID
    title = "KR447 FM transceiver (integral antenna, or the conducted U.FL variant)"
    triggers = ("radio_build",)
    needs = BUILD.needs
    serves = BUILD.serves
    plane_nets = ("GND", None)
    layer_policy = LayerPolicy(allowed=(4,), default=4, reason=PLANE_REASON)

    # ------------------------------------------------------------------ selection

    def triggered_by(self, ir: CircuitIR, inputs: dict[str, DesignInput]) -> bool:
        return read_radio_build(ir)[0] in VARIANTS

    def refusals(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str]) -> list[MissingInformation]:
        """The family's closed world (a requirement no transceiver build serves names the builds that do) and the FM-only rule."""
        build = read_radio_build(ir)[0] or RADIO_BUILD
        out: list[MissingInformation] = []
        modulation, _why = read_modulation(ir)
        if modulation is not None and modulation != BUILD.modulation:
            why = f"modulation {modulation}: the KR 447 MHz licence-exempt class is FM (F3E) [UNVERIFIED: 「무선설비규칙」]; radio_build={build} builds an FM radio only"
            out.append(MissingInformation(key="modulation", required=False, rationale=why,
                                          question=f"{why}; no template design was proposed. State modulation=FM or choose another design."))
        for r in unserved_requirements(ir, self):
            canon = canonical_key(r.key) or r.key
            why = f"{r.id} ({requirement_text(r)}): {unserved_message(canon, build)}"
            out.append(MissingInformation(key=r.key, required=False, rationale=why,
                                          question=f"{why}; no template design was proposed. Start the project of the build that serves it, or leave this requirement out of the {build} board."))
        return out

    @staticmethod
    def out_of_range(inputs: dict[str, DesignInput]) -> str | None:
        """A stated input outside what the blocks are designed for, or a carrier that is no channel of the (unverified) raster; ``None`` otherwise."""
        f = inputs.get("carrier_frequency")
        if f is not None:
            why = raster_refusal(float(f.traced.value), float(PROFILE_BY_KEY["kr447.band_low"].value), float(PROFILE_BY_KEY["kr447.band_high"].value),
                                 float(PROFILE_BY_KEY["kr447.channel_raster"].value))
            if why is not None:
                return f"{f.requirement.id}: {why}"
        for key, (lo, hi), why in _RANGES:
            inp = inputs.get(key)
            if inp is not None and not lo <= float(inp.traced.value) <= hi:
                return f"{key} {float(inp.traced.value):.12g} {inp.traced.unit} ({inp.requirement.id}) is outside {lo:.12g}..{hi:.12g} {inp.traced.unit}: {why}"
        for key in ("link_range", "field_strength_limit", "erp", "eirp"):
            inp = inputs.get(key)
            if inp is not None and not float(inp.traced.value) > 0:
                return f"{key} {float(inp.traced.value):.12g} {inp.traced.unit} ({inp.requirement.id}) must be positive"
        return None

    # ------------------------------------------------------------------ build

    def build(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str], library: KicadLibrary, *, confirmed: bool) -> Plan:
        t = self.id
        variant, why_build = read_radio_build(ir)
        if variant not in VARIANTS:
            plan = Plan(template=t, title=self.title)
            return _refused(plan, f"radio_build {variant!r} is not a transceiver build ({', '.join(VARIANTS)}){': ' + why_build if why_build else ''}")
        plan = Plan(template=t, title=BUILDS[variant].title)
        stated = {k: inputs[k] for k in self.serves if k in inputs}
        why = self.out_of_range(inputs)
        if why is not None:
            plan.inputs = stated
            return _refused(plan, why)
        present = present_keys(ir, inputs)
        missing = [k for k in ("carrier_frequency", "input_voltage") if k not in present]
        if missing:
            _missing_inputs(plan, f"The {BUILDS[variant].title} template", missing, unusable, examples={"carrier_frequency": "447.5625 MHz", "input_voltage": "7.4 V"})
        modulation, mod_why = read_modulation(ir)
        if modulation is None:
            plan.questions.append(MissingInformation(key="modulation", required=mod_why is None, rationale="template input",
                                                     question=mod_why or "The KR447 FM transceiver template needs modulation: answer modulation=fm (the KR 447 MHz class is FM [UNVERIFIED])"))
            if not missing:
                return _refused(plan, f"modulation missing{': ' + mod_why if mod_why else ''}")
        if plan.questions or plan.notes:
            return plan
        if modulation != BUILD.modulation:
            return _refused(plan, f"modulation {modulation!r}: the KR 447 MHz licence-exempt class is FM telephony [UNVERIFIED: 「무선설비규칙」]; this radio is FM only")
        tot = TX_TIMEOUT_KEY in inputs
        z0_value = float(inputs["system_impedance"].traced.value) if "system_impedance" in inputs else 50.0
        r_ant = inputs.get("antenna_impedance")
        with_match = variant == VARIANT_ANTENNA and r_ant is not None and abs(float(r_ant.traced.value) - z0_value) > 1e-9 * z0_value
        plan.inputs = stated
        profile = profile_choices(t, confirmed)
        ctx = BlockContext(ir=ir, library=library, template_id=t, confirmed=confirmed, inputs=inputs, shared={c.key: tr for c, tr in profile})
        results: list[BlockResult] = []
        try:
            for blk, prefix in transceiver_blocks(variant, tot=tot, with_match=with_match):
                if blk.id not in REGIONS:
                    raise TemplateRefusal(f"block {blk.id!r} has no floorplan region in REGIONS")
                r = blk.build(ctx, prefix)
                results.append(r)
                ctx.shared.update({k: v for k, v in r.params.items() if k not in ctx.shared})
            # the antenna end's cascade fixture (ant_end): the networks the board joins with no resistive node between them, as one fixture
            ant_end = ant_end_result(ctx, results, with_match=with_match)
            board_values = self._board_values(ctx, [r.block_id for r in results], variant)
            merged = merge_results([board_values, *results, ant_end], t)
        except TemplateRefusal as e:
            return _refused(plan, str(e))
        components, report = exclude_floating(merged.components, merged.nets, merged.stimuli, t)
        bad = [f for f in report.unresolved if not f.ref.startswith("RV")]  # a pot's wiper keeps its DC path through the pot's own resistors
        if bad:
            return _refused(plan, "a multi-terminal part has a node nothing else simulated touches: " + "; ".join(f.reason for f in bad))
        components = self._trace_obw(ctx, components)
        rf_blocks = [RFBlock(id=r.block_id, title=r.title, refs=[c.ref for c in r.components], chain=list(r.chain), shield_ref=r.shield_ref,
                             region=self._region(merged, r.block_id), ports=list(r.ports)) for r in results]
        try:
            rf = RFDesign(blocks=rf_blocks, networks=list(merged.networks), frequency_plan=list(merged.plan_lines), lab_items=list(merged.lab_items),
                          rails=list(merged.rails), model_values=sorted(merged.model_keys), profile_keys=profile_keys())
            plan.keepouts = self._keepouts(ctx, merged, variant)
        except (ValueError, TemplateRefusal) as e:
            return _refused(plan, f"the RF design refuses the composition: {e}")
        plan.choices = list(merged.choices)
        plan.computed = list(merged.computed)
        plan.parts = [part_line(merged.placed[c.ref]) for c in components if c.ref in merged.placed]
        plan.nets = [_net_line(n) for n in merged.nets]
        sim = SimulationSetup(stimuli=list(merged.stimuli), analyses=list(merged.analyses), expectations=list(merged.expectations))
        from ai_eda.design.rf.t_tx_exciter import Kr447TxExciterTemplate

        plan.simulation = Kr447TxExciterTemplate._simulation_lines(merged, sim, components, report)
        plan.notes.extend(merged.notes)
        if variant == VARIANT_ANTENNA and r_ant is not None and not with_match:
            plan.notes.append(f"antenna_impedance ({r_ant.requirement.id}) equals the system impedance: the low-pass feeds the antenna directly, no match is built")
        if variant == VARIANT_CONDUCTED and r_ant is not None:
            plan.notes.append(f"antenna_impedance ({r_ant.requirement.id}) is recorded but no match is fitted on the conducted variant: J1001 is the system-impedance "
                              "port the instruments expect (the transceiver build carries the match)")
        topology = Topology(
            name=BUILDS[variant].title, domains=[CircuitDomain.RF, CircuitDomain.ANALOG, CircuitDomain.POWER],
            rationale=("KR 447 MHz licence-exempt class FM transceiver (every KR number UNVERIFIED): "
                       + ("integral straight quarter-wave wire antenna" if variant == VARIANT_ANTENNA else "U.FL conducted antenna port (never an antenna)")
                       + " -> harmonic low-pass -> PIN T/R switch; TX: TCXO f_c / 12 -> two buffered PM tanks (indirect FM from the integrated, limited, "
                       "splatter-filtered audio) -> x3 / x2 / x2 -> band-pass -> PHA-1 -> MMZ09332BT1 -> load-line match -> switch; RX: switch -> "
                       "band-pass / BFR92 LNA / band-pass -> ADEX-10 (LO1 = f_c - 21.4 MHz from a x12 chain) -> diplexer -> post-amp -> crystal ladder -> "
                       "SA605D -> squelch -> LM386; PTT sequencing, TX interlock and the low-pack inhibit in hardware, no firmware. No IC has a model: "
                       "the passive networks are the RF fixtures, judged under confirmed model values"),
            provenance=ctx.provenance(f"selected by radio_build = {variant}"),
            blocks=[TopologyBlock(id=r.block_id, function=r.title, domain=CircuitDomain.RF if r.block_id in RF_BLOCK_IDS else CircuitDomain.ANALOG,
                                  component_refs=[c.ref for c in r.components], input_nets=[p.net for p in r.ports if p.direction == "in"],
                                  output_nets=[p.net for p in r.ports if p.direction == "out"], provenance=ctx.provenance(f"block {r.block_id}")) for r in results],
        )
        plan.changes = _changes(t, BUILDS[variant].title, topology, components, merged.nets, dict(merged.params), sim,
                                [*merged.constraints, *self._constraints(ctx, variant)])
        plan.changes.append(DesignChange(description="RF design: blocks, fixture networks, frequency plan, lab items, rails", target="rf", operation="set", payload=rf,
                                         rationale=f"template {t}: {len(rf.networks)} fixture network(s), {len(rf.frequency_plan)} plan row(s), {len(rf.lab_items)} lab item(s)"))
        return plan

    # ------------------------------------------------------------------ build steps

    def _board_values(self, ctx: BlockContext, block_ids: list[str], variant: str) -> BlockResult:
        """The rows of the board, not of a block: the profile placeholders, the floorplan, the band, OBW, the antenna-port sensitivity, the
        radiated numbers and the gated plan row."""
        b = BlockBuilder(ctx, "board", "board values")
        for ch, traced in profile_choices(ctx.template_id, ctx.confirmed):
            b.result.choices.append(ch)
            b.result.params[ch.key] = traced
        for bid in block_ids:
            for axis, value in zip(("x", "y", "w", "h"), REGIONS[bid]):
                what = {"x": "left edge", "y": "top edge", "w": "width", "h": "height"}[axis]
                b.choice(f"floor.{bid}.{axis}", value, "mm", f"floorplan region of block {bid} ({_REGION_WHAT[bid]}): {what} (mm, board frame)")
        if variant == VARIANT_ANTENNA:  # the antenna band (a keep-out of the antenna build only)
            b.choice("floor.ant_band.h", BAND_H_MM, "mm", ("height of the antenna band along the top edge: no parts, tracks, vias or pads but the antenna's feed and "
                                                          "ANT_FEED; the planes stay allowed up to their edge inset (the monopole's counterpoise reaches the feed)"))
        sh = ctx.shared
        try:
            # occupied bandwidth (numbers, no verdict: the KR test modulation is unverified)
            dev_in = ctx.inputs.get("frequency_deviation")
            if dev_in is not None:
                dev_key, dev = DEVIATION_PARAM, copy_input(b, DEVIATION_PARAM, dev_in.traced)
            else:
                dev_key, dev = "kr447.max_deviation", b.result.params["kr447.max_deviation"]
            for tag, f in (("1k", 1000.0), ("3k", 3000.0)):
                fm = b.choice(f"tx.obw.f_{tag}", f, "Hz", f"the single test tone of the {tag} occupied-bandwidth number (which tone the KR test method prescribes is [UNVERIFIED])")
                calc(b, f"tx.obw99_{tag}", lambda fm=fm, tag=tag: radio.fm_obw99(dev, fm, (dev_key, f"tx.obw.f_{tag}")))
            # the receive sensitivity at the antenna port (an estimate: decision 2A)
            front = b.choice("trx.rx.front_loss", 1.3, "dB", (
                "receive loss in front of RX_RF: the harmonic low-pass (1.045 dB on the design's Q-40 deck) and the T/R switch in RX (about 0.25 dB); a choice "
                "for the budget - the fixtures bound the two (lpf s21 at least -1.5 dB, trsw rx s21 at least -0.5 dB)"))
            nf = calc(b, "trx.nf.total", lambda: radio.db_sum(front, sh["fe.nf.total"], ("trx.rx.front_loss", "fe.nf.total")))
            calc(b, "trx.sensitivity", lambda: sensitivity(sh["fe.t0"], sh["rf.if_bw"], nf, sh["fe.snr"], ("fe.t0", "rf.if_bw", "trx.nf.total", "fe.snr")))
            # radiated numbers (only from stated inputs; the verdicts are the lab's)
            p_dbm = calc(b, "ant.p_tx_dbm", lambda: w_to_dbm(sh["pa.p_out"], ("pa.p_out",)))
            g_in, d_in = ctx.inputs.get("antenna_gain"), ctx.inputs.get("link_range")
            eirp_dbm = None
            if g_in is not None:
                g = copy_input(b, "ant.g", g_in.traced)
                l_feed = b.choice("ant.l_feed", 1.5, "dB", ("loss between the PA's match and the antenna in TX: the T/R switch (about 0.35 dB) and the low-pass (about "
                                                            "1.05 dB at Q_u 40), rounded up - a choice for the radiated numbers"))
                eirp_dbm = calc(b, "ant.eirp_dbm", lambda: eirp(p_dbm, g, l_feed, ("ant.p_tx_dbm", "ant.g", "ant.l_feed")))
                calc(b, "ant.erp_dbm", lambda: eirp_to_erp(eirp_dbm, ("ant.eirp_dbm",)))
            if d_in is not None:
                d = copy_input(b, "ant.d", d_in.traced)
                calc(b, "ant.fspl", lambda: fspl(sh["rf.f_c"], d, ("rf.f_c", "ant.d")))
                if eirp_dbm is not None:
                    eirp_w = calc(b, "ant.eirp_w", lambda: dbm_to_w(eirp_dbm, ("ant.eirp_dbm",)))
                    calc(b, "ant.e_at_range", lambda: field_strength(eirp_w, d, ("ant.eirp_w", "ant.d")))
        except (KeyError, ValueError) as e:
            raise TemplateRefusal(f"board values: {e}") from e
        f_c, f_tx = sh.get("rf.f_c"), sh.get("tx.f3")
        if f_c is not None and f_tx is not None:
            b.result.plan_lines.append(PlanLine(
                id="tx_ref_on_rx_channel", kind="gated", f_hz=f_tx, ref_hz=f_c, points_to=["spice.tx_rail_off_rx", "rf.lab.tr_sequencing"],
                note=("N x f_T lands exactly on the receive channel: the TX reference and its multiplier chain must be unpowered in RX (the TX rail off "
                      "while PTT is released)")))
        return b.done()

    @staticmethod
    def _region(merged: BlockResult, bid: str) -> RFRegion | None:
        vals = {axis: merged.params.get(f"floor.{bid}.{axis}") for axis in ("x", "y", "w", "h")}
        if any(v is None for v in vals.values()):
            return None
        return RFRegion(**vals)  # type: ignore[arg-type]

    @staticmethod
    def _trace_obw(ctx: BlockContext, components: list) -> list:
        """A confirmed ``occupied_bandwidth`` is served by the parts that set it: the splatter low-pass (the audio bandwidth) and the deviation
        trim (the deviation) - the parts the Carson / Bessel bandwidth follows from."""
        inp = ctx.inputs.get("occupied_bandwidth")
        if inp is None:
            return components
        rid = inp.requirement.id
        out = []
        for c in components:
            if ("Sallen-Key" in c.description or "deviation trim" in c.description) and rid not in c.serves_requirements:
                c = c.model_copy(update={"serves_requirements": [*c.serves_requirements, rid]})
            out.append(c)
        return out

    @staticmethod
    def _keepouts(ctx: BlockContext, merged: BlockResult, variant: str) -> list[Keepout]:
        """``ant_band`` (the top band, every copper layer) and ``trx_bcu`` (no B.Cu tracks under the switch / low-pass region) - see the module docstring."""
        width = max(float(merged.params[f"floor.{bid}.x"].value) + float(merged.params[f"floor.{bid}.w"].value) for bid in REGIONS)
        feed = FEED_REF[variant]
        tx, ty, tw, th = (float(merged.params[f"floor.{TRX_ID}.{a}"].value) for a in ("x", "y", "w", "h"))
        band = merged.params.get("floor.ant_band.h")
        band_h = float(band.value) if band is not None else BAND_H_MM
        prov = choice_provenance(ctx.template_id, f"antenna band keep-out: the top {band_h:g} mm of the {width:g} mm board (floor.ant_band.h)", ctx.confirmed)
        prov_b = choice_provenance(ctx.template_id, "B.Cu track keep-out under the trx region (floor.trx.*)", ctx.confirmed)
        out: list[Keepout] = []
        if variant == VARIANT_ANTENNA:  # the band belongs to the radiator: the conducted variant (no antenna) has none
            out.append(Keepout(
                id="ant_band", layers=["*.Cu"], rect=Traced(value=[0.0, 0.0, width, band_h], unit="mm", provenance=prov),
                forbids=["tracks", "vias", "pads", "footprints"], allowed_refs=[feed], allowed_nets=[FEED_NET],
                reason=("antenna band at the top edge: nothing near the base of the quarter-wave monopole but the antenna wire's feed and the ANT_FEED "
                        "line; the ground planes stay (they are the monopole's counterpoise and reach the feed)"),
                provenance=ctx.provenance("keep-out ant_band (kr447 design §2.5, critic2 layout (a))")))
        return [
            *out,
            Keepout(id="trx_bcu", layers=["B.Cu"], rect=Traced(value=[tx, ty, tw, th], unit="mm", provenance=prov_b), forbids=["tracks"],
                    allowed_nets=[GROUND_NET],
                    reason=("no B.Cu tracks under the quarter-wave and harmonic low-pass inductors (a track below couples into their fields); the whole trx "
                            "region, because the packer decides where in it the inductors sit; GND is exempt so that every shunt part's ground via "
                            "(c.kr447.rf_ground_vias) has its B.Cu pad there - the router's track ban also bars other nets' via pads"),
                    provenance=ctx.provenance("keep-out trx_bcu (kr447 design §2.5)")),
        ]

    @staticmethod
    def _constraints(ctx: BlockContext, variant: str) -> list[Constraint]:
        s = ctx.provenance
        out = [
            Constraint(id="c.kr447.kc_before_transmission", kind=ConstraintKind.REGULATORY, target="*", provenance=s("KR 447 MHz licence-exempt class"),
                       description=("every kr447.* number (band, raster, 0.5 W, F3E, deviation, OBW, tolerance, spurious limits, integral antenna, time-out) is an "
                                    "UNVERIFIED placeholder until 「신고하지 아니하고 개설할 수 있는 무선국용 무선기기」 and 「무선설비규칙」 are grounded; KC "
                                    "conformity assessment (전파법 제58조의2) is needed before any transmission, a self-built unit included")),
            Constraint(id="c.kr447.rf_ground_vias", kind=ConstraintKind.RF, target="*", provenance=s("stopband of the fixtures"),
                       description=("every shunt part of the tanks, band-passes, matches, the T/R switch and the harmonic low-pass needs a via to the ground plane "
                                    "at its pad: at 447 MHz 1 mm of track to ground is about 1 nH, and the fixtures (no ground-return inductance) hold only then")),
            Constraint(id="c.kr447.pa_pd_polarity", kind=ConstraintKind.ELECTRICAL, target="PA_PD", provenance=s("PA power-down polarity"),
                       description=("PA_PD high = PA powered down (from the library pin name POWER_DOWN [UNVERIFIED: NXP MMZ09332B datasheet]); check the "
                                    "polarity before the first key-up")),
            Constraint(id="c.kr447.tr_sequencing", kind=ConstraintKind.ELECTRICAL, target="*", provenance=s("T/R switch sequencing"),
                       description=("at PTT release the PA supply PA_5V must be off before the PIN bias from TX_5V decays, or the LNA sees the carrier "
                                    "(pa_supply_off_first proves the PA_5V side in the deck: the PA powers down at the release edge and draws next to "
                                    "nothing, so the PTT block's active discharge Q205 empties PA_5V; with ideal rails the deck cannot show that Q201's "
                                    "body diode holds PA_5V at most a diode drop above TX_5V or that POWER_DOWN, pulled up to TX_3V3, deasserts once "
                                    "TX_3V3 collapses - the hold-up is the lab item tr_sequencing)")),
        ]
        if variant == VARIANT_CONDUCTED:
            out.append(Constraint(id="c.kr447.conducted_sample", kind=ConstraintKind.REGULATORY, target="*", provenance=s("conducted variant"),
                                  description=("the transceiver_conducted build transmits only conducted, through J1001 into an attenuator or a dummy load; it is "
                                               "never fitted with an antenna (the profile's integral-antenna condition is the product's). Whether KC accepts such a "
                                               "sample, and whether a conducted bench test needs a permit, is [UNVERIFIED: RRA test method; 전파법 제58조의3]")))
        else:
            out.append(Constraint(id="c.kr447.integral_antenna", kind=ConstraintKind.REGULATORY, target="*", provenance=s("integral antenna"),
                                  description=("ANT1 is soldered to its feed pad: an integral, non-detachable antenna (kr447.antenna [UNVERIFIED]); no antenna "
                                               "connector on the product")))
        return out

    # ------------------------------------------------------------------ board

    #: the TX supply path the POWER_PA class sizes (0.6 A, above the 441 mA TX budget)
    POWER_PA_CURRENT_A = 0.6
    POWER_PA_TEMP_RISE_C = 10.0

    def si_declarations(self, ir: CircuitIR, ctx: BoardContext) -> SIDeclarations:
        """Every stage's classes on the composed board (each net in one class, the first that names it): ``RF50_H`` (harmonics before the
        low-pass), ``RF50`` (system-impedance lines at f_c), ``RF_ANT`` (the antenna side of a match), ``RF50_TX`` (the TX lines at
        ``tx.z_mid``: TX_RAW to the PA input), ``IF50`` / ``IF_NODE`` / ``IF_HIZ`` (the IF chain), ``RF_TANK_LO`` / ``RF_TANK_HI`` (resonator nodes), ``TX_LUMPED``
        (lumped TX nodes) and ``POWER_PA`` (IPC-2221 width for 0.6 A)."""
        from ai_eda.design.board import SIDeclarations
        from ai_eda.design.rf.t_rx_backend import if_backend_hiz_nets
        from ai_eda.design.rf.t_rx_frontend import IF50_NETS, IF_NODE_NETS, RF50_NETS, TANK_HI_NETS, TANK_LO_NETS
        from ai_eda.design.rf.blocks.pa import pa_net_classes
        from ai_eda.design.rf.blocks.trx import trx_net_classes
        from ai_eda.design.rf.blocks.tx_chain import CHAIN_NET_CLASSES, DRIVER_NET_CLASSES, MOD_NET_CLASSES
        from ai_eda.tools.calc.basic import ipc2221_width_for_current

        out = SIDeclarations()
        p = ctx.params
        z0, z_mid, f_c, f_3, if1, f2 = (p.get(k) for k in ("rf.z0", "tx.z_mid", "rf.f_c", "lpf.f_3", "rf.if1", "lo.f2"))
        if None in (z0, z_mid, f_c, f_3, if1, f2):
            return out
        with_match = "trx.match.l" in p
        trx = trx_net_classes(with_match=with_match)
        pa = pa_net_classes(with_lpf=False, out_net=TX_NET)
        tx: dict[str, list[str]] = {}
        for classes in (MOD_NET_CLASSES, CHAIN_NET_CLASSES, DRIVER_NET_CLASSES, pa):  # the blocks' own class lists: every name is a net of the board
            for name, nets in classes.items():
                tx.setdefault(name, []).extend(nets)
        taken: set[str] = set()

        def members(*groups) -> list[str]:
            """The nets of ``groups`` no earlier class took (one class per net: the first that names it)."""
            got = []
            for g in groups:
                for n in g:
                    if n not in taken:
                        taken.add(n)
                        got.append(n)
            return got

        tol = ctx.choice("si.rf50_z0_tol", 0.10, None, "tolerance of the RF50 / RF50_H / RF50_TX / IF50 / RF_ANT classes' impedance (10 %)")
        prov = ctx.structural
        rf50_h = members(tx["RF50_H"], trx["RF50_H"])
        rf50 = members(RF50_NETS, trx["RF50"])
        rf_ant = members(trx.get("RF_ANT", []))
        rf50_tx = members(tx["RF50"])
        if50 = members(IF50_NETS)
        if_node = members(IF_NODE_NETS)
        if_hiz = members(if_backend_hiz_nets(p, PREFIXES[IF_BLOCK_ID].net_prefix))
        tank_lo = members(TANK_LO_NETS)
        tank_hi = members(TANK_HI_NETS)
        lumped = members(tx["TX_LUMPED"], trx["TX_LUMPED"])
        out.classes += [
            NetClass(name="RF50_H", nets=rf50_h, target_z0_ohm=z0, z0_tol_rel=tol, rf_frequency_hz=f_3,
                     description="the PA's match output, the T/R switch and the low-pass input: they carry the harmonics before the low-pass (rf frequency 3 f_c)",
                     provenance=prov("harmonic-carrying lines before the low-pass")),
            NetClass(name="RF50", nets=rf50, target_z0_ohm=z0, z0_tol_rel=tol, rf_frequency_hz=f_c,
                     description="the system-impedance lines at the carrier: the low-pass output / antenna feed, the switch's RX side, the front end, mixer and LO lines",
                     provenance=prov("50 ohm lines at the carrier")),
            NetClass(name="RF50_TX", nets=rf50_tx, target_z0_ohm=z_mid, z0_tol_rel=tol, rf_frequency_hz=f_c,
                     description="the TX lines from the band-pass output TX_RAW through the pads and the driver to the PA input", provenance=prov("TX 50 ohm lines")),
            NetClass(name="IF50", nets=if50, target_z0_ohm=z0, z0_tol_rel=tol, rf_frequency_hz=if1,
                     description="the mixer's IF port, the post-amp and IF1 into the IF back-end: 50 ohm lines at IF1 (electrically short)", provenance=prov("50 ohm IF lines")),
            NetClass(name="IF_NODE", nets=if_node, rf_frequency_hz=if1, description="the diplexer's internal nodes: lumped at IF1, no impedance target",
                     provenance=prov("diplexer nodes")),
            NetClass(name="IF_HIZ", nets=if_hiz, description=("the crystal ladder, SA605 mixer, IF2, limiter, quadrature and LO2 nodes: high-impedance lumped nodes at "
                                                             "21.4 MHz / 450 kHz, no impedance target, no driven edge"),
                     provenance=prov("IF back-end high-impedance nodes")),
            NetClass(name="RF_TANK_LO", nets=tank_lo, rf_frequency_hz=f2,
                     description="the LO TCXO and the first two multiplier stages' nodes (35.5 / 106.5 / 213.1 MHz): lumped, keep short against lambda_g / 10",
                     provenance=prov("multiplier tank nodes below 300 MHz")),
            NetClass(name="RF_TANK_HI", nets=tank_hi, rf_frequency_hz=f_c,
                     description="the UHF resonator nodes (the LO x12 collector, the LO and front-end band-passes): lumped, keep short against lambda_g / 10",
                     provenance=prov("UHF resonator nodes")),
            NetClass(name="TX_LUMPED", nets=lumped, description=("the PM tanks, followers, TX multiplier and resonator nodes, the PA output at its load line and the "
                                                                "low-pass's inner nodes: lumped nodes, no line impedance applies - keep them short"),
                     provenance=prov("TX lumped nodes")),
        ]
        if rf_ant:
            out.classes.append(NetClass(name="RF_ANT", nets=rf_ant, target_z0_ohm=p["trx.ant.r"], z0_tol_rel=tol, rf_frequency_hz=f_c,
                                        description="the antenna side of the L-match at the confirmed antenna resistance", provenance=prov("antenna feed line")))
        i = ctx.choice("si.power_pa_current", self.POWER_PA_CURRENT_A, "A", (
            "current the TX supply path's minimum width is sized for: 0.6 A, above the 441 mA TX budget [UNVERIFIED: the budget is an estimate]"), param=True)
        dt = ctx.choice("si.power_pa_temp_rise", self.POWER_PA_TEMP_RISE_C, "degC", "temperature rise the POWER_PA class's IPC-2221 width is sized for", param=True)
        if ctx.stackup is not None:
            layer = ctx.stackup.copper_layer("F.Cu")
            if layer is not None:
                w = ctx.computed_param("si.w_power_pa", ipc2221_width_for_current(i, dt, layer.thickness_um,
                                                                                    ("si.power_pa_current", "si.power_pa_temp_rise", "pcb.stackup.copper[F.Cu].thickness_um")))
                power = members(["VBAT", "VBAT_F", "V_SYS", "V_TX", "TX_5V", "PA_5V"])
                out.classes.append(NetClass(name="POWER_PA", nets=power, min_width_mm=w, power_current_a=i, power_temp_rise_c=dt,
                                            description="the pack-to-PA supply path: at least the IPC-2221 width for 0.6 A", provenance=prov("TX supply path minimum width")))
        out.classes = [c for c in out.classes if c.nets]
        out.lines.append("RF50 / RF50_H / RF50_TX / IF50 / RF_ANT: 50 ohm (or antenna) lines over In1.Cu; the other classes are lumped nodes without a line "
                         "impedance; si.rf_length needs ir.si.rf_length_fraction, which the board module does not write yet")
        return out

    # ------------------------------------------------------------------ report views

    def theory(self, ir: CircuitIR) -> list[TheorySection]:
        p = lambda k: _p(ir, k)  # noqa: E731
        q = lambda k, u: quantity(p(k), u)  # noqa: E731
        variant = VARIANT_CONDUCTED if any(c.ref == "J1001" for c in ir.components) else VARIANT_ANTENNA
        profile_rows = "\n".join(f"| `{e.key}` | {e.value if isinstance(e.value, str) else quantity(float(e.value), e.unit)} | {e.document} |" for e in PROFILE)
        lq, cq = p("trx.lq.l"), p("trx.lq.c")
        z0, f_c = p("rf.z0"), p("rf.f_c")
        w = 2.0 * math.pi * f_c if f_c else None
        x_l = w * lq if w and lq else None
        x_c = 1.0 / (w * cq) if w and cq else None
        length = p("trx.ant.length")
        wave = 299792458.0 / f_c if f_c else None
        sens, nf, nf_fe, front = p("trx.sensitivity"), p("trx.nf.total"), p("fe.nf.total"), p("trx.rx.front_loss")
        match = p("trx.match.l") is not None
        corners = [(p(f"floor.{bid}.x"), p(f"floor.{bid}.y"), p(f"floor.{bid}.w"), p(f"floor.{bid}.h")) for bid in REGIONS]
        known = [c for c in corners if None not in c]
        board_w = max(c[0] + c[2] for c in known) if known else None  # type: ignore[operator]
        board_h = max(c[1] + c[3] for c in known) if known else None  # type: ignore[operator]
        if variant == VARIANT_ANTENNA:
            ant_text = (
                f"ANT1 은 급전 패드에 납땜하는 곧은 λ/4 도선(결정 5A)입니다. 길이 l = vf · c0 / (4 f_c) (`calc.rf.quarter_wave`) = {number(p('trx.ant.vf'), 3)} × "
                f"{quantity(wave / 4.0 if wave else None, 'm')} = {quantity(length, 'm')}. 속도 계수 vf 는 선택값이며, 실제 공진 길이는 60 mm 폭 접지면(대응극) 위에서 "
                "VNA 로 재는 실험 항목(`rf.lab.antenna_length`)입니다. 급전 패드는 배치기의 2 mm 여백 때문에 판 위쪽 가장자리에서 약 3-4 mm 안쪽, 평면 가장자리 "
                "근처에 놓이며(critic2 의 배치 (a)), 도선의 처음 몇 mm 가 평면 위를 지나는 영향은 계산기에 없습니다.\n\n"
                + ("안테나 임피던스가 확인되어 L 정합(`calc.rf.lmatch.lowpass.*`: 직렬 L 은 낮은 저항 쪽, 병렬 C 는 높은 저항 쪽)을 둡니다: "
                   f"L1003 = {q('trx.match.l', 'H')}, C1006 = {q('trx.match.c', 'F')}, 안테나 저항 {q('trx.ant.r', 'ohm')}. 픽스처 `ant_match`: S21 ≥ −0.5 dB, S11 ≤ −15 dB."
                   if match else
                   "`antenna_impedance` 가 확인되지 않아 저역통과 필터가 시스템 임피던스로 안테나를 바로 급전합니다. 정합은 VNA 측정 뒤 새 `antenna_impedance` 로 하는 "
                   "사람의 설계 변경입니다(`rf.lab.antenna_match`)."))
        else:
            ant_text = ("이 변형(`transceiver_conducted`)은 안테나 자리에 U.FL J1001 을 둡니다. 벤치 시험과 KC 전도 시료용이며 **절대로 안테나를 달지 않습니다**. "
                        "J1001 은 계측기가 기대하는 시스템 임피던스 포트이므로 안테나 정합은 두지 않습니다(정합은 일체형 안테나 변형에만). 블록 영역 배치는 "
                        "제품과 같게 유지해 전도 시료가 제품의 기판 배치를 대표하도록 했고, 방사체가 없으므로 안테나 띠 keep-out 은 두지 않습니다. KC 가 임시 안테나 "
                        "커넥터 시료를 받는지는 검증되지 않았습니다.")
        return [
            TheorySection("개요: 447 MHz FM 무전기 (5단계 조합)", (
                "KR 447 MHz 면허 불요 FM 무전기 계열의 5단계 기판입니다. 1-4단계 시험 기판의 블록을 같은 빌더와 같은 번호(전원 1xx, PTT 2xx, 송신 오디오 3xx, "
                "수신 오디오 4xx, IF 백엔드 5xx, 수신 프런트엔드 6xx, LO 7xx, 송신 체인 8xx, PA 9xx)로 한 기판에 조합하고, T/R 스위치·고조파 저역통과 필터·안테나 "
                "정합(10xx)과 안테나 급전부를 더했습니다. 조합한 IR 은 해시가 달라 모든 검사를 처음부터 다시 실행합니다(시험 기판의 PASS 는 이월되지 않음).\n\n"
                f"변형: **{variant}** - " + ("일체형 직선 λ/4 도선 안테나 ANT1." if variant == VARIANT_ANTENNA else "안테나 대신 U.FL J1001 (전도 시료, 안테나 금지).") + "\n\n"
                "**모든 KR 규제 수치(`kr447.*`)는 법령 원문으로 근거를 두지 않은 자리표시값(UNVERIFIED)이며 `rf.regulatory_profile` 은 PASS 가 되지 않습니다.** "
                "자작품이라도 송신 전에 KC 적합성평가(전파법 제58조의2)가 필요합니다. RF 성능(출력, 편이, 점유 대역폭, 스퓨리어스, 감도, 선택도, ERP)은 실험실에서만 "
                "판정되며(`rf.lab.*`), 여기의 PASS 는 '확인된 모델값 아래에서의 회로망·원리 판정'입니다.\n\n"
                "| 규제 프로파일 키 | 값 (검증되지 않음) | 확인할 문서 |\n|---|---|---|\n" + profile_rows
            )),
            TheorySection("송수신 전환과 모드 게이팅", (
                f"송신 기준 f_T = f_c / N 의 N 배(= {q('tx.f3', 'Hz')})는 수신 채널 f_c 와 정확히 같습니다. 그래서 수신 중에는 TX 기준 발진기와 체배 체인이 꺼져 있어야 "
                "하며(TX 레일 차단), 이것은 `rf.freq_plan` 의 게이트 행 `tx_ref_on_rx_channel` 이 설계 덱의 `tx_rail_off_rx` 와 실험 항목 `tr_sequencing` 을 가리키는 "
                "방식으로 기록됩니다(자체 판정 없음). PTT 를 놓을 때는 PA 전원(PA_5V)이 PIN 바이어스(TX_5V)보다 먼저 꺼져야 LNA 가 반송파를 보지 않습니다: 해제 순간 PA 는 "
                "POWER_DOWN 으로 꺼져 전류를 거의 쓰지 않으므로, PA_PD 로 켜지는 능동 방전(Q205 / R224)이 PA_5V 를 비웁니다. 설계 덱은 그 PA_5V 쪽"
                "(`pa_supply_off_first`)만 보여 주며(레일이 이상 전원이라 Q201 의 바디 다이오드와 TX_3V3 붕괴 뒤 POWER_DOWN 해제는 보이지 않음), 실제 유지 시간은 "
                "실험 항목입니다."
            )),
            TheorySection("PIN T/R 스위치 (직렬-병렬, 집중정수 λ/4)", (
                "송신 쪽 직렬 PIN D1001, 공통 노드에서 수신 쪽으로 가는 집중정수 λ/4 π 회로(C1001 - L1001 - C1002), 수신 끝의 병렬 PIN D1002 로 이루어집니다.\n\n"
                "    L = Z0 / ω,  C = 1 / (ω Z0)            (`calc.rf.quarter_wave_lumped.l` / `.c`)\n"
                "    R_bias = (V_TX5 − 2 V_F) / I_bias      (`calc.led.R`, 다이오드 두 개의 순방향 전압을 하나의 선택값으로)\n\n"
                f"| 항목 | 값 |\n|---|---|\n| f_c / Z0 | {quantity(f_c, 'Hz')} / {quantity(z0, 'ohm')} |\n| L1001 | {quantity(lq, 'H')} (X_L = {quantity(x_l, 'ohm')}) |\n"
                f"| C1001 = C1002 | {quantity(cq, 'F')} (X_C = {quantity(x_c, 'ohm')}) |\n| R1001 | {q('trx.pin.r', 'ohm')} (I = {q('trx.pin.i_bias', 'A')}, 2 V_F = {q('trx.pin.v_f_pair', 'V')}) |\n"
                f"| 초크 L1002 / DC 차단 | {q('trx.l_choke', 'H')} / {q('trx.c_block', 'F')} |\n\n"
                "송신: TX_5V 가 켜지면 바이어스 전류가 R1001 → L1002 → D1001 → L1001 → D1002 → GND 로 흘러 두 다이오드가 모두 켜집니다. D1001 이 반송파를 공통 노드로 "
                "통과시키고, D1002 가 수신 끝을 단락하며, λ/4 구간이 그 단락을 공통 노드에서 개방으로 바꿔 송신 전력이 수신 쪽으로 가지 않습니다. 수신: 두 다이오드가 "
                "꺼져 λ/4 구간이 50 Ω 선로처럼 공통 노드의 신호를 RX_RF 로 전달하고, D1001 의 C_off 가 송신 쪽을 격리합니다.\n\n"
                "픽스처 `trsw`: 상태 tx (두 다이오드 = `model.pin.r_on`) 에서 TX → 공통 S21 ≥ −0.5 dB, TX → RX ≤ −25 dB; 상태 rx (= `model.pin.c_off`) 에서 공통 → RX "
                "≥ −0.5 dB, 공통 → TX ≤ −20 dB. 인덕터에는 `model.l_q.uhf` 의 직렬 손실, 바이어스 급전(R1001, L1002)은 교류 접지된 레일 포트로 포함됩니다. 설계 덱의 "
                "`pin_bias` 는 0 Ω 링크 R1002 를 흐르는 전류를 `calc.led.I` 공칭값 ±20 % 로 판정합니다(`model.diode` 는 PIN 모델이 아님). +27 dBm 에서의 손실·격리·"
                "발열은 실험 항목 `trsw_power` 입니다."
            )),
            TheorySection("고조파 저역통과 필터와 안테나 급전", (
                "4단계의 7차 0.1 dB 체비셰프 저역통과 필터(리플 모서리 "
                f"{q('lpf.f_edge', 'Hz')}, `calc.rf.lpf.*`)를 T/R 스위치 뒤로 옮겼습니다(설계 §1.2): 스위치 자체의 고조파도 걸러지고, 수신은 그 손실을 치릅니다. "
                "픽스처 `lpf`: f_c 에서 S21 ≥ −1.5 dB, S11 ≤ −15 dB, 2 f_c 에서 ≤ −45 dB, 3 f_c 에서 ≤ −60 dB (병렬 소자마다 접지면 비아가 있는 배치에서만 성립).\n\n"
                "안테나 쪽 회로망들은 기판에서 저항성 노드 없이 바로 이어집니다: TX_RF 에서 부하선 정합의 DC 차단과 스위치의 TX DC 차단, LPF_IN 에서 스위치의 공통 DC 차단과 "
                "저역통과 필터의 첫 병렬 C, RX_RF 에서 스위치의 RX DC 차단과 fe_bpf2 의 입력 탭(안테나 정합이 있으면 ANT_PORT 에서 필터와 정합). 각 픽스처는 그곳을 "
                "시스템 임피던스로 가정했으므로, 조합 수준의 캐스케이드 픽스처 `ant_end` 가 네(다섯) 회로망의 부품 전체를 PA 부하선, 안테나 급전, LNA 베이스 사이에서 "
                "스위치의 tx / rx 상태로 판정합니다: TX 에서 S21 ≥ " + q("ant_end.tx_s21_min", "dB") + " (정합·스위치·필터 한계의 합), 부하선에서 S11 ≤ "
                + q("pa.match.s11_max", "dB") + ", 2 f_c / 3 f_c ≤ " + q("lpf.h2_max", "dB") + " / " + q("lpf.h3_max", "dB") + ", TX → LNA ≤ " + q("trx.tx_iso_max", "dB")
                + "; RX 에서 S21 ≥ " + q("ant_end.rx_s21_min", "dB") + " (필터·스위치 한계와 fe_bpf2 의 정확한 S21 의 합), 영상(f_c 기준) ≤ " + q("ant_end.rx_image_max", "dB")
                + ", RX → PA ≤ " + q("trx.rx_iso_max", "dB") + ".\n\n"
                + ant_text
            )),
            TheorySection("수신 감도와 방사 수치 (추정, 판정 아님)", (
                "수신 감도는 프런트엔드의 Friis 예산(`fe.nf.total`)에 RX_RF 앞의 손실(저역통과 + 스위치, 선택값 `trx.rx.front_loss`)을 더한 안테나 포트 추정입니다:\n\n"
                "    NF_ant = L_front + NF_fe          (`calc.rf.db_sum`)\n"
                "    S = kTB + NF_ant + SNR            (`calc.rf.sensitivity`)\n\n"
                f"L_front = {quantity(front, 'dB')}, NF_fe = {quantity(nf_fe, 'dB')}, NF_ant = {quantity(nf, 'dB')}, 감도 약 {quantity(sens, 'dBm')} "
                "(결정 2A 의 약 −113 dBm; 가정한 소자 수치 아래의 추정). 실제 12 dB SINAD 감도는 실험 항목 `rf.lab.sensitivity` 입니다.\n\n"
                f"송신 전력 {q('pa.p_out', 'W')} = {q('ant.p_tx_dbm', 'dBm')} (`calc.rf.w_to_dbm`). "
                + (f"안테나 이득 {q('ant.g', 'dBi')} 와 급전 손실 {q('ant.l_feed', 'dB')} 에서 EIRP = P + G − L_feed (`calc.rf.eirp`) = {q('ant.eirp_dbm', 'dBm')}, "
                   f"ERP = {q('ant.erp_dbm', 'dBm')} (`calc.rf.eirp_to_erp`). " if p("ant.eirp_dbm") is not None else
                   "`antenna_gain` 이 주어지지 않아 EIRP / ERP 는 계산하지 않았습니다. ")
                + (f"거리 {q('ant.d', 'm')} 에서 자유공간 손실 {q('ant.fspl', 'dB')} (`calc.rf.fspl`)"
                   + (f", 전계 강도 {q('ant.e_at_range', 'V/m')} (`calc.rf.field_strength`). " if p("ant.e_at_range") is not None else ". ")
                   if p("ant.fspl") is not None else "`link_range` 가 주어지지 않아 경로 손실은 계산하지 않았습니다. ")
                + "모두 판정 없는 수치이며 방사 판정은 실험실의 것입니다."
            )),
            TheorySection("기판 배치(floorplan)와 keep-out", (
                f"기판 폭은 설계의 60 mm 이고 안테나 끝이 위쪽입니다. 설계의 60 × 145 mm 윤곽에는 조합된 부품 {len(ir.components)} 개가 들어가지 않습니다(엄밀한 "
                "하한): 캔 밖 부품과 캔의 외곽을 배치기의 1 mm 간격만큼 키운 면적 합이 약 8.5 × 10³ mm² 인데, 윤곽이 2 mm 가장자리 여백 안에 남기는 면적은 "
                "같은 간격을 더해도 57 × 142 = 8.1 × 10³ mm² 입니다(안테나 띠는 빼지도 않은 값). 또 1-4단계에서 차폐 캔이 네 개(BMI-S-103 셋, BMI-S-105 "
                "하나)가 되었습니다. BMI-S-103 두 개를 나란히 두려면 여백을 포함해 60.04 mm 가 필요하므로 캔마다 한 "
                f"줄을 쓰고, 윤곽은 영역의 외접 사각형인 {number(board_w, 4)} × {number(board_h, 4)} mm 입니다(`floor.*` 선택값).\n\n"
                + ("keep-out `ant_band`: 위쪽 `floor.ant_band.h` mm, 모든 동박층에서 부품·패드·트랙·비아 금지(안테나 급전부와 ANT_FEED 만 예외). 접지면은 허용해 "
                   "모노폴의 대응극이 급전점까지 옵니다. " if variant == VARIANT_ANTENNA else
                   "전도 변형에는 방사체가 없으므로 안테나 띠 keep-out 이 없습니다(영역 배치는 제품과 같음). ")
                + "`trx_bcu`: trx 영역 아래 B.Cu 트랙 금지(λ/4 와 저역통과 인덕터). GND 만 예외입니다: 라우터는 트랙 금지를 비아의 B.Cu 패드에도 적용하므로 "
                "예외가 없으면 병렬 소자의 접지 비아(`c.kr447.rf_ground_vias`)를 영역 안에 둘 수 없고, 두 내층이 모두 GND 평면이라 B.Cu 의 GND 는 접지 위의 "
                "접지입니다. 컴파일된 기판에는 예외 부분을 잘라 낸 규칙 영역으로 쓰이며 예외는 IR 검사 `pcb.keepout` 만 압니다. 이 배치는 RF 품질 판정이 "
                "아니며(라우터는 RF 를 모름). 배치기(`placement.rf_floorplan` 0.2)는 MAX9814(U301, 0.4 mm 피치)와 PA(U901, 0.5 mm 피치) 둘레에 팬아웃 여유를 "
                "남겨 다른 부품을 패드에서 2.2 mm 떨어뜨리고(결정 2A), 라우터(`routing.maze`)는 두 PHA-1(SOT-89-3)의 사용자 정의 패드를 앵커와 도형의 상자로 "
                "읽고 U301·U901 의 0.25 mm 폭 패드를 탈출 스텁으로 0.2 mm 격자에 잇습니다. 0.6 의 첫 통과가 U301 의 일부 핀에 탈출을 주지 못하면(U302.2 의 "
                "접지 비아가 U301 의 탈출 영역에 들어감) 정적 단계를 0.7 로 한 번 더 돌려 U301 의 팬아웃 여유에서 다른 부품의 평면 비아를 빼고 탈출의 출구를 "
                "여유의 가장자리까지 잇게 하며, 넷이 100개를 넘는 큰 기판은 협상을 최대 150회까지 합니다. 2026-09-30 두 빌드의 배치된 기판에서 라우터만 돌려 "
                "측정: 211 넷 모두 배선, 123회에서 합법(40회 상한이었다면 충돌 넷 5·6개가 남았음). 같은 날 전체 파이프라인에서는 긴 넷 다섯 개(PTT_ACTIVE, "
                "PA_PD, MUTE, SQ_SET, PM_DRIVE)가 Z50 으로 승격되고 한 번의 재배선이 98·104회에서 합법으로 끝나 적용되었으며, si.impedance.Z50 이 "
                "PASS, FAIL 인 검사는 없고 RELEASE 는 NOT_VERIFIED 입니다. 마이크 MK301 은 스루홀 POM-2244P-C3310-2-R 입니다(결정 1A; 심볼 핀 1 '-' / 2 '+' 가 "
                "패드 1 / 2 에 붙지만 어느 단자가 케이스(음극)인지는 PUI Audio 데이터시트로 확인하지 않았습니다(검증되지 않음)). 앞서 쓰던 "
                "SMT 캡슐 CUI_CMC-4013-SMT 는 패드 2 가 링 모양 패드 1 안에 있어 패드 안 비아 없이는 MIC_P 를 F.Cu 로 꺼낼 수 없었습니다.\n\n"
                "사용자가 만지는 부품의 가장자리 배치는 설계(§2.5: 음량·스퀠치 가변저항과 마이크, 팩·스피커 커넥터를 아래쪽 가장자리에)와 다릅니다. 영역 "
                "배치기는 각 블록의 부품을 영역 왼쪽 위부터 채우므로, SW201(PTT)만 설계대로 왼쪽 가장자리에 있고 RV401(음량)은 아래쪽 가장자리에서 44.2 mm, "
                "RV402(스퀠치)는 29.9 mm 위(수신 오디오 영역), MK301 은 왼쪽 가장자리에서 아래쪽으로부터 171 mm 위(LO 체인 옆 송신 오디오 영역), J401(스피커)은 "
                "아래쪽에서 52 mm 위, J101(팩)은 왼쪽 가장자리에서 아래쪽으로부터 19 mm 위에 놓입니다(두 빌드 같음). 이 부품들의 가장자리 배치는 KiCad 에서 "
                "손으로 옮기는 일로 남겨 둡니다."
            )),
            TheorySection("모델 값과 검증의 한계", (
                "이 기판의 PASS 는 **확인된 모델 값 아래의 회로망·원리 판정**입니다(실제 부품의 측정이 아님): 인덕터 Q, 수정·바랙터·PIN 다이오드 모델, 포트 저항, "
                "범용 트랜지스터·다이오드·MOSFET 카드, 연산증폭기 매크로. `rf.model_grounding` 은 이 값들이 근거를 얻을 때까지 NOT_VERIFIED 입니다. IC(SA605, "
                "ADEX-10, PHA-1, PA, TCXO, MAX9814, LM386, 비교기, 논리, LDO)는 모두 넷리스트에서 제외됩니다. 잡음지수, S 파라미터, 대신호 PA 동작, 오디오 시간 동안의 "
                "UHF 과도 해석은 없습니다. RELEASE 는 최선의 경우에도 NOT_VERIFIED 입니다.\n\n"
                "아래 절들은 1-4단계 시험 기판 템플릿의 이론 설명을 이 조합 기판의 수치로 다시 쓴 것입니다(각 절 제목 앞의 단계 표시)."
            )),
            *self._stage_sections(ir),
        ]

    @staticmethod
    def _stage_sections(ir: CircuitIR) -> list[TheorySection]:
        """The stage boards' theory sections on this IR's numbers, without their bench-board overviews and model-limit sections (this board's own
        sections cover both), each title marked with its stage."""
        from ai_eda.design.rf.t_audio_ptt import KR447_AUDIO_PTT
        from ai_eda.design.rf.t_rx_backend import KR447RxBackendTemplate
        from ai_eda.design.rf.t_rx_frontend import KR447RxFrontendTemplate
        from ai_eda.design.rf.t_tx_exciter import KR447_TX_EXCITER

        out: list[TheorySection] = []
        for tag, template in (("1단계 오디오·PTT", KR447_AUDIO_PTT), ("2단계 IF 백엔드", KR447RxBackendTemplate()),
                              ("3단계 수신 프런트엔드", KR447RxFrontendTemplate()), ("4단계 송신 여진기", KR447_TX_EXCITER)):
            for s in template.theory(ir):
                if s.title.startswith(_SKIPPED_STAGE_SECTIONS):
                    continue
                body = s.body
                for old, new in _STAGE_TEXT_ON_THIS_BOARD:
                    body = body.replace(old, new)
                out.append(TheorySection(f"[{tag}] {s.title}", body))
        return out

    def theory_figures(self, ir: CircuitIR) -> list[Figure]:
        """The quarter-wave wire's length against its velocity factor (the antenna build), then the stage boards' figures on this IR's numbers."""
        from ai_eda.design.rf.t_audio_ptt import KR447_AUDIO_PTT
        from ai_eda.design.rf.t_rx_backend import KR447RxBackendTemplate
        from ai_eda.design.rf.t_rx_frontend import KR447RxFrontendTemplate
        from ai_eda.design.rf.t_tx_exciter import KR447_TX_EXCITER

        out: list[Figure] = []
        f_c, vf = _p(ir, "rf.f_c"), _p(ir, "trx.ant.vf")
        if f_c and vf is not None:
            xs = _lin_grid(0.80, 1.00, 81)
            ys = [x * 299792458.0 / (4.0 * f_c) * 1000.0 for x in xs]
            caption = (f"'고조파 저역통과 필터와 안테나 급전' 절의 λ/4 도선 길이 l = vf c0 / (4 f_c) (`calc.rf.quarter_wave`, f_c = {quantity(f_c, 'Hz')}); 점 = 선택값 "
                       f"vf {number(vf, 3)}. {THEORY_CURVE_NOTE}")
            out.append(_curve_figure("theory_ant_length", "λ/4 도선 길이 대 속도 계수", caption, [("λ/4 길이", xs, ys)], x_label="속도 계수 vf",
                                     y_label="길이 (mm)", markers=[(vf, vf * 299792458.0 / (4.0 * f_c) * 1000.0, "vf")]))
        seen = {f.id for f in out}
        for template in (KR447_AUDIO_PTT, KR447RxBackendTemplate(), KR447RxFrontendTemplate(), KR447_TX_EXCITER):
            for fig in template.theory_figures(ir):
                if fig.id not in seen:
                    seen.add(fig.id)
                    out.append(fig)
        return out

    def part_notes(self, ir: CircuitIR) -> dict[str, PartNote]:
        """Each block's parts get the notes of the stage board that owns the block (the trx / antenna parts their own)."""
        from ai_eda.design.rf.t_audio_ptt import stage1_part_notes, stage1_refs
        from ai_eda.design.rf.t_rx_backend import KR447RxBackendTemplate
        from ai_eda.design.rf.t_rx_frontend import KR447RxFrontendTemplate
        from ai_eda.design.rf.t_tx_exciter import KR447_TX_EXCITER

        blocks = {b.id: set(b.refs) for b in (ir.rf.blocks if ir.rf is not None else [])}
        owned = lambda *ids: set().union(*(blocks.get(i, set()) for i in ids))  # noqa: E731
        stage1 = stage1_refs(ir)
        out: dict[str, PartNote] = stage1_part_notes(ir, stage1)
        for template, ids in ((KR447RxBackendTemplate(), (IF_BLOCK_ID,)), (KR447RxFrontendTemplate(), (FRONTEND_ID, MIXER_ID, LO_CHAIN_ID, LO_BUFFER_ID)),
                              (KR447_TX_EXCITER, (MOD_ID, TX_CHAIN_ID, DRIVER_ID, PA_ID))):
            refs = owned(*ids)
            notes = template.part_notes(ir)
            out.update({r: n for r, n in notes.items() if r in refs})
        p = lambda k: _p(ir, k)  # noqa: E731
        for c in ir.components:
            if c.ref not in owned(TRX_ID, ANTENNA_ID):
                continue
            d = c.description
            if c.ref == "ANT1":
                out[c.ref] = PartNote("일체형 λ/4 도선 안테나 (결정 5A)", f"급전 패드(SolderWire 0.5 mm²)에 납땜하는 곧은 도선, 길이 {quantity(p('trx.ant.length'), 'm')} (vf {number(p('trx.ant.vf'), 3)})",
                                      ["0.5 mm² 단선, 길이는 VNA 로 맞춤", "분리할 수 없게 고정(일체형 안테나 조건 [UNVERIFIED])", "60 mm 폭 접지면이 대응극"],
                                      [unverified("일체형 헬리컬 / 칩 안테나", "계산기 밖의 설계: 사람의 설계 변경과 VNA 측정 필요")])
            elif c.ref == "J1001":
                out[c.ref] = PartNote("전도 안테나 포트 (U.FL, 전도 시료 전용)", "감쇠기 / 더미 로드 전용 - 안테나 금지", ["50 Ω", "0.5 W 에서 U.FL 정격 확인"],
                                      [unverified("Hirose U.FL-R-SMT-1(10)")])
            elif "PIN diode" in d:
                out[c.ref] = PartNote("T/R 스위치 PIN 다이오드", "직렬(D1001, 송신 쪽) / 병렬(D1002, 수신 끝); 픽스처에서 R_on / C_off 모델값",
                                      [f"R_on ≤ {number(p('model.pin.r_on'), 3)} Ω @ 10 mA", f"C_off ≤ {quantity(p('model.pin.c_off'), 'F')}", "+27 dBm 처리, SOD-323"],
                                      [unverified("Infineon BAR64-03W"), unverified("Skyworks SMP1302", "패키지·핀 배열 확인")])
            elif "quarter-wave section" in d:
                out[c.ref] = PartNote("λ/4 집중정수 소자", "`calc.rf.quarter_wave_lumped.*` 계산값 그대로", [f"값 {c.value}", "NP0 / 고 Q 인덕터(0604HQ 급)", "병렬 소자마다 접지면 비아"],
                                      [unverified("Coilcraft 0604HQ / Murata GJM 0402")])
            elif "harmonic low-pass" in d:
                out[c.ref] = PartNote("고조파 저역통과 필터 소자 (스위치 뒤)", "7차 0.1 dB 체비셰프 (`calc.rf.lpf.*`)", [f"값 {c.value}", "NP0, 고 Q 인덕터", "병렬 소자마다 접지면 비아"],
                                      [unverified("Mini-Circuits LFCN-490 (라이브러리에 있음)", "삽입손실·저지대역 확인")])
            elif "antenna L-match" in d:
                out[c.ref] = PartNote("안테나 L 정합 소자", "`calc.rf.lmatch.lowpass.*` (확인된 antenna_impedance)", [f"값 {c.value}", "고 Q, NP0"],
                                      [unverified("Coilcraft 0604HQ / Murata GJM 0402")])
            elif "PIN bias" in d or "0 ohm PIN" in d:
                out[c.ref] = PartNote("PIN 바이어스 부품", "TX_5V → R1002(0 Ω 링크) → R1001 → L1002 → 스위치", [f"값 {c.value}", "초크 자기공진 > f_c"],
                                      [unverified("0402 1 % / 0603 권선 초크")])
            elif "DC block" in d:
                out[c.ref] = PartNote("T/R 스위치 DC 차단", "바이어스를 스위치 안에 가둠", [f"값 {c.value}", "NP0 0402"], [unverified("Murata GJM 0402")])
            else:
                out[c.ref] = PartNote(f"{c.ref}: {d}", "라이브러리에서 핀 이름으로 배선", ["핀 배열과 정격을 데이터시트로 확인"], [])
        return out



KR447_TRANSCEIVER = Kr447TransceiverTemplate()

__all__ = [
    "BAND_H_MM",
    "BOARD_W_MM",
    "BUILD",
    "KR447_TRANSCEIVER",
    "PLANE_REASON",
    "PREFIXES",
    "REGIONS",
    "TEMPLATE_ID",
    "V_IN_RANGE",
    "VARIANTS",
    "Kr447TransceiverTemplate",
    "transceiver_blocks",
]
