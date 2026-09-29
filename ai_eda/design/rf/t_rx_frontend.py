"""The ``kr447_rx_frontend`` template (``radio_build = rx_frontend``): the stage-3 bench board of the KR 447 MHz FM receiver (kr447 design §2.3).

Invariant: a pure function of the confirmed requirements, the library on
disk and the block builders - nothing is guessed, nothing is grounded that is
not. The board is the composition (:func:`~ai_eda.design.rf.blocks.base.merge_results`)
of four RF blocks and the stage board's own bench parts:

* :class:`~ai_eda.design.rf.blocks.lo_chain.LoChainBlock` (``7xx``, under the
  can SH701): TCXO -> x3 -> x2 -> x2 with the double-tuned tanks
  ``lo_tank1`` / ``lo_tank2`` after the first two stages and the
  4-resonator LO band-pass ``lo_bpf`` after the last (one network from the
  x12 collector to the PHA-1's input ``LO1_RAW``);
* :class:`~ai_eda.design.rf.blocks.lo_chain.LoBufferBlock` (``7xx``): the
  PHA-1 buffer and the pad ``lo_pad`` into the mixer's LO port;
* :class:`~ai_eda.design.rf.blocks.rx_frontend.RxFrontendBlock` (``6xx``,
  under the can SH601): ``fe_bpf2`` -> BFR92 LNA -> ``fe_bpf3``;
* :class:`~ai_eda.design.rf.blocks.rx_frontend.RxMixerBlock` (``6xx``):
  ADEX-10 -> ``diplexer`` -> BFR92 IF1 post-amp, and the sensitivity budget;
* the bench ports :class:`BenchRfInBlock` (``J601``, the RF input U.FL on
  the left edge) and :class:`BenchIfOutBlock` (``J602``, the IF1 output
  U.FL on the right edge) - the transceiver (part P13) composes the RF
  blocks without them.

The supply is a *companion*: the RX power section of part P9
(:func:`default_companions`: ``PowerBlock(modes=("rx",))``, references
``1xx``, internal nets ``PWR_*``; kr447 design §2.1), which reads
``input_voltage`` (6.4-8.4 V: above the confirmed pack cut-off, at most a
full 2S pack; no TX inhibit on a receive-only board) and writes the rail
levels ``power.rx_5v`` / ``power.rx_3v3`` the blocks read from
:attr:`BlockContext.shared`; its ``main_switch_on`` row joins the design
deck. A template built with no companions **refuses** instead of building a
board without a supply; :class:`BenchSupplyBlock` - one bench header
standing in for it (RX_5V and RX_3V3 from the bench, honest about being a
stand-in: ``input_voltage`` is then served by nothing) - is what most tests
compose (``KR447RxFrontendTemplate(companions=(BenchSupplyBlock(),))``).

Selection and the closed world (kr447 design §2.0): selected only by the
confirmed categorical requirement ``radio_build = rx_frontend``
(:meth:`triggered_by`); it needs ``carrier_frequency`` (which must be a
channel of the unverified KR 447 MHz raster, else a refusal naming the
channel list - a band such as "447 MHz 대역" is not a carrier) and
``input_voltage`` (read by the RX power section); it serves the keys of
``family.BUILDS["rx_frontend"]`` - ``modulation`` optional (if stated it
must read ``fm``), ``system_impedance`` (the RF / IF ports' Z0),
``frequency_tolerance`` (the TCXO's stability spec), ``rx_sensitivity``
(named by the lab item beside the budget estimate, never refused, never
PASS) - and refuses every other confirmed design requirement with the
family's sentence naming the builds that serve it. A stated ``pcb_layers``
other than 4 refuses before anything is asked (the layer policy: the RF
lines need a reference plane).

What the plan carries: every ``kr447.*`` regulatory placeholder as a choice
whose description ends ``[UNVERIFIED: ...]`` (``ir.rf.profile_keys``; never
grounded, ``rf.regulatory_profile`` never PASSes), every block choice and
``model.*`` value (``ir.rf.model_values``: UNVERIFIED, ``rf.model_grounding``
NOT_VERIFIED), the floorplan regions (``floor.<block>.x`` / ``.y`` / ``.w`` /
``.h`` mm, choices; ``placement.rf_floorplan`` packs each block into its
region, the cans first, and the outline is their bounding box), the RF
design ``ir.rf`` (blocks with their interface ports, eight fixture networks,
the frequency plan, lab items) and the signal-integrity classes
(:meth:`si_declarations`).

Expected statuses on this machine (ngspice-42, the packed KiCad 10.0.6
libraries, no kicad-cli; measured by ``tests/test_kr447_rx_frontend.py``):
every ``spice.rf.*`` row of the eight networks PASS as "a network verdict
under confirmed model values (not a measured part)"; ``spice.ic_*`` (the
five bias points) PASS under ``model.npn``; ``rf.freq_plan`` NOT_VERIFIED
(its margin rows PASS; its response rows - image, half-IF, the four LO-spur
responses - carry no verdict of their own and point to lab items);
``rf.model_grounding`` / ``rf.regulatory_profile`` / ``rf.lab.*``
NOT_VERIFIED; ``block.interface.*`` PASS (IR arithmetic). The floorplan
places 106 parts on a 71 x 64 mm outline with the bench supply header,
118 on 71 x 82 mm with the RX power section below (both cans BMI-S-103: the
13.3 mm BMI-S-102 holds only 16 of the front end's 25 parts). Not asserted
by the tests, seen once through the whole pipeline: ``routing.maze``
refuses the board (the PHA-1's ``SOT-89-3`` footprint has a ``custom`` pad
shape), so the board stays placement-only: ``pcb.routing.connectivity``
FAILs on the nets whose pads are all known and have no copper (the nets
holding the custom pad are NOT_VERIFIED rows), ``pcb.routing.clearance`` has
no copper to compare, ``pcb.keepout`` / ``pcb.silk.clearance`` /
``domain.rf.impedance`` are NOT_VERIFIED, and RELEASE FAILs; with the bench
stand-in ``input_voltage`` is served by nothing, so
``review.requirements_vs_ir`` FAILs there too (the default companion serves
it).
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Iterable

from ai_eda.ir import Block as TopologyBlock
from ai_eda.ir import CircuitDomain, CircuitIR, MissingInformation, NetClass, NetKind, SimulationSetup, Stimulus, StimulusKind, Topology, Traced
from ai_eda.ir.rf import RFBlock, RFDesign, RFPort, RFRegion
from ai_eda.tools.calc import radio
from ai_eda.tools.kicad.library import KicadLibrary

from ai_eda.design.base import (
    Choice,
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
from ai_eda.design.rf.blocks.base import Block, BlockBuilder, BlockContext, BlockPrefix, BlockResult, exclude_floating, merge_results
from ai_eda.design.rf.blocks.lo_chain import BUFFER_ID, CHAIN_ID, LoBufferBlock, LoChainBlock
from ai_eda.design.rf.blocks.power import PACK_CUTOFF_V, PACK_MAX_V
from ai_eda.design.rf.blocks.rx_frontend import FRONTEND_ID, MIXER_ID, RxFrontendBlock, RxMixerBlock, exclude, requirement_ids
from ai_eda.design.rf.family import BUILDS, unserved_message
from ai_eda.design.rf.models import MODEL_VERDICT
from ai_eda.design.rf.parts import PlacedPart
from ai_eda.design.rf.profile import profile_choices, profile_keys, raster_refusal

if TYPE_CHECKING:
    from ai_eda.design.board import BoardContext, SIDeclarations
    from ai_eda.report.figures import Figure

#: the radio_build this template builds
RADIO_BUILD = "rx_frontend"
#: the family's entry (needs / serves / board text)
BUILD = BUILDS[RADIO_BUILD]
#: why only 4 layers (kr447 design §2.0, decision 7A)
PLANE_REASON = "the RF lines need a reference plane (decision 7A: 4 layers)"
#: the stage board's own bench blocks
BENCH_IN_ID, BENCH_OUT_ID, BENCH_SUPPLY_ID = "bench_rf_in", "bench_if_out", "bench_io"
#: how each block is named on the board: references re-based by hundreds, internal nets prefixed (``power``: part P9's companion)
PREFIXES: dict[str, BlockPrefix] = {
    CHAIN_ID: BlockPrefix(700, ""),
    BUFFER_ID: BlockPrefix(700, ""),
    FRONTEND_ID: BlockPrefix(600, ""),
    MIXER_ID: BlockPrefix(600, ""),
    BENCH_IN_ID: BlockPrefix(600, ""),
    BENCH_OUT_ID: BlockPrefix(600, ""),
    BENCH_SUPPLY_ID: BlockPrefix(0, ""),
    "power": BlockPrefix(100, "PWR_"),
}
#: the floorplan region of each block (x, y, w, h in mm from the board's top-left corner): choices ``floor.<block>.*``; the board is their
#: bounding box. The cans (BMI-S-103, 27.52 mm) need 29.5 mm regions; the RX power section is a strip along the bottom edge (71 x 18 mm: its
#: 13 parts, the power switch's 12.7 mm body the tallest).
REGIONS: dict[str, tuple[float, float, float, float]] = {
    BENCH_IN_ID: (0.0, 12.0, 8.0, 8.0),
    FRONTEND_ID: (8.0, 0.0, 31.0, 32.0),
    MIXER_ID: (39.0, 0.0, 24.0, 26.0),
    BENCH_OUT_ID: (63.0, 9.0, 8.0, 8.0),
    CHAIN_ID: (8.0, 32.0, 31.0, 32.0),
    BUFFER_ID: (39.0, 32.0, 24.0, 20.0),
    BENCH_SUPPLY_ID: (63.0, 34.0, 8.0, 14.0),
    "power": (0.0, 64.0, 71.0, 18.0),
}
#: the pack voltages the RX power section is designed for: above the confirmed cut-off (``power.pack_cutoff_v``), at most a full 2S pack
V_IN_RANGE: tuple[float, float] = (PACK_CUTOFF_V, PACK_MAX_V)
#: the order the blocks are built in: the supply first (its rail levels are shared), the LO chain before the front end (whose filters probe the
#: LO-spur responses), the front end before the mixer (whose budget reads the filters' losses), the bench ports last (they read the plan values)
RF_BLOCKS: tuple[type[Block], ...] = (LoChainBlock, LoBufferBlock, RxFrontendBlock, RxMixerBlock)


# --------------------------------------------------------------------------- the stage board's bench parts


class BenchSupplyBlock(Block):
    """One bench header in place of the RX power section: ``J1`` RX_5V / RX_3V3 / GND from a bench supply (a stand-in, see the module docstring)."""

    id = BENCH_SUPPLY_ID
    title = "bench supply header: RX_5V and RX_3V3 (stand-in for the RX power section)"
    interface_nets = ("RX_5V", "RX_3V3")

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        v5 = b.choice("power.rx_5v", 5.0, "V", "RX_5V from the bench supply on J1 (stand-in for the RX power section's LP38693DT-5.0 output)")
        v3 = b.choice("power.rx_3v3", 3.3, "V", "RX_3V3 from the bench supply on J1 (stand-in for the RX power section's LP5907-3.3 output)")
        j = b.part("header_3", "J1", "Conn_01x03", "bench supply header: RX_5V, RX_3V3, GND in (stand-in for the RX power section)")
        b.net("RX_5V", NetKind.POWER, j.at("Pin_1"), "bench supply RX_5V")
        b.net("RX_3V3", NetKind.POWER, j.at("Pin_2"), "bench supply RX_3V3")
        b.net("GND", NetKind.GROUND, j.at("Pin_3"), "ground")
        exclude(b, "J1", "bench header: a connector, no electrical model (the rails are the ideal sources VRX5V / VRX3V3)")
        for sid, net, value in (("RX5V", "RX_5V", v5), ("RX3V3", "RX_3V3", v3)):
            b.result.stimuli.append(Stimulus(id=sid, source="voltage", net=net, reference_net="GND", kind=StimulusKind.DC, value=value,
                                             provenance=ctx.provenance(f"{net}: the bench supply (an ideal source; the header J1 has no model)")))
        b.result.ports = [RFPort(name="RX_5V", net="RX_5V", kind="rail", voltage_v=v5, direction="out"),
                          RFPort(name="RX_3V3", net="RX_3V3", kind="rail", voltage_v=v3, direction="out")]
        b.result.chain = ["J1"]
        b.result.notes.append("bench_io: a header stands in for the RX power section (part P9): input_voltage is read by that block, not by this stand-in")
        return b.done()


class _BenchPort(Block):
    """A bench U.FL on one interface net (``In`` on the net, ``Ext`` on GND), with its 50 ohm port."""

    ref: str
    net: str
    what: str
    port_direction: str
    frequency_key: str

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, (self.net,))
        z0, f = ctx.shared.get("rf.z0"), ctx.shared.get(self.frequency_key)
        if z0 is None or f is None:
            raise TemplateRefusal(f"block {self.id} reads rf.z0 and {self.frequency_key} from the RF blocks built before it (BlockContext.shared)")
        serves = requirement_ids(ctx, "system_impedance")
        j = b.part("coax_ufl", self.ref, "U.FL", self.what, serves)
        exclude(b, self.ref, "U.FL connector: excluded; a fixture port of the system impedance where a network ends on it")
        b.net(self.net, NetKind.RF, j.at("In"), self.what, serves=serves)
        b.net("GND", NetKind.GROUND, j.at("Ext"), "ground")
        b.result.ports = [RFPort(name=self.net, net=self.net, kind="port", z0_ohm=z0, frequency_hz=f, direction=self.port_direction)]  # type: ignore[arg-type]
        b.result.chain = [self.ref]
        return b.done()


class BenchRfInBlock(_BenchPort):
    """``J1`` (``J601``): the RF input U.FL on ``RX_RF`` (a signal generator on the bench)."""

    id = BENCH_IN_ID
    title = "bench RF input U.FL (RX_RF)"
    ref, net, what, port_direction, frequency_key = "J1", "RX_RF", "RF input U.FL (447 MHz, signal generator)", "out", "rf.f_c"


class BenchIfOutBlock(_BenchPort):
    """``J2`` (``J602``): the IF1 output U.FL on ``IF1`` (to the rx_backend board by cable)."""

    id = BENCH_OUT_ID
    title = "bench IF1 output U.FL (IF1)"
    ref, net, what, port_direction, frequency_key = "J2", "IF1", "IF1 output U.FL (21.4 MHz, to the IF back-end board)", "in", "rf.if1"


def default_companions() -> tuple[Block, ...]:
    """The RX power section (part P9) the board composes: ``PowerBlock(modes=("rx",))``."""
    from ai_eda.design.rf.blocks.power import PowerBlock

    return (PowerBlock(modes=("rx",)),)


# --------------------------------------------------------------------------- the template


def _refused(plan: Plan, why: str) -> Plan:
    plan.notes.append(f"template {plan.template} not proposed: {why}")
    return plan


class KR447RxFrontendTemplate(Template):
    """``radio_build = rx_frontend``: RF U.FL -> BPF / LNA / BPF -> ADEX-10 -> diplexer -> post-amp -> IF1 U.FL, with the LO1 chain (module docstring)."""

    id = BUILD.template_id
    title = BUILD.title
    triggers = ("radio_build",)
    needs = BUILD.needs
    serves = BUILD.serves
    plane_nets = ("GND", None)
    layer_policy = LayerPolicy(allowed=(4,), default=4, reason=PLANE_REASON)

    def __init__(self, companions: Iterable[Block] | None = None) -> None:
        self._companions = None if companions is None else tuple(companions)

    def companions(self) -> tuple[Block, ...]:
        return default_companions() if self._companions is None else self._companions

    # ------------------------------------------------------------------ selection

    def triggered_by(self, ir: CircuitIR, inputs: dict[str, DesignInput]) -> bool:
        return read_radio_build(ir)[0] == RADIO_BUILD

    def refusals(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str]) -> list[MissingInformation]:
        """The closed world with the family's sentence: every unserved confirmed design requirement names the builds that serve it."""
        out: list[MissingInformation] = []
        for r in unserved_requirements(ir, self):
            canon = canonical_key(r.key) or r.key
            why = f"{r.id} ({requirement_text(r)}): {unserved_message(canon, RADIO_BUILD)}"
            out.append(MissingInformation(
                key=r.key, required=False, rationale=why,
                question=f"{why}; no template design was proposed. Start the project of the build that serves it, or leave this requirement out of the {RADIO_BUILD} board.",
            ))
        return out

    # ------------------------------------------------------------------ build

    def build(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str], library: KicadLibrary, *, confirmed: bool) -> Plan:
        t = self.id
        plan = Plan(template=t, title=self.title)
        present = present_keys(ir, inputs)
        missing = [k for k in self.needs if k not in present]
        if missing:
            for k in missing:
                if k in unusable:
                    plan.notes.append(f"{k} not usable: {unusable[k]}")
                elif k == "carrier_frequency":
                    plan.questions.append(MissingInformation(key=k, rationale="template input", question=(
                        f"The {self.title} template needs carrier_frequency in Hz: the channel of the KR 447 MHz raster the receiver is built for "
                        "(e.g. carrier_frequency=\"447.5625 MHz\"; a band such as '447 MHz 대역' is not a channel) [UNVERIFIED channel list]")))
                else:
                    plan.questions.append(MissingInformation(key=k, rationale="template input", question=(
                        f"The {self.title} template needs {k} in V (the pack voltage the RX power section regulates): answer {k}=<value V> (e.g. {k}=\"7.4 V\")")))
            return _refused(plan, f"input(s) {missing} missing")
        modulation, why_mod = read_modulation(ir)
        if why_mod is not None:
            return _refused(plan, f"modulation not usable: {why_mod}")
        if modulation is not None and modulation != BUILD.modulation:
            return _refused(plan, f"modulation {modulation!r}: the KR 447 MHz licence-exempt class is FM telephony [UNVERIFIED: 「무선설비규칙」]; this receiver is built for FM")
        profile = profile_choices(t, confirmed)
        shared: dict[str, Traced] = {c.key: tr for c, tr in profile}
        f_c = float(inputs["carrier_frequency"].traced.value)
        band = [float(shared[k].value) for k in ("kr447.band_low", "kr447.band_high", "kr447.channel_raster")]
        why_raster = raster_refusal(f_c, *band)
        if why_raster is not None:
            return _refused(plan, why_raster)
        v_in = inputs["input_voltage"]
        if not V_IN_RANGE[0] <= v_in.traced.value <= V_IN_RANGE[1]:
            return _refused(plan, (f"input_voltage {v_in.traced.value:.12g} V ({v_in.requirement.id}) is outside {V_IN_RANGE[0]:.12g}..{V_IN_RANGE[1]:.12g} V: "
                                   f"the 2S Li-ion pack is used only above the confirmed cut-off {PACK_CUTOFF_V:.12g} V (the minimum input of the LP38693DT-5.0), "
                                   f"and a full 2S pack is {PACK_MAX_V:.12g} V (the P-FET gates see the whole pack [UNVERIFIED: AOS AO3401A datasheet])"))
        companions = self.companions()
        if not companions:
            return _refused(plan, ("the RX power section (kr447 design §2.1, part P9) is not composed: it reads input_voltage and writes the RX_5V / RX_3V3 "
                                   "levels every stage of this board is biased from, and a board without it would leave input_voltage unserved "
                                   "(default_companions() composes it)"))
        plan.inputs = {k: inputs[k] for k in self.serves if k in inputs}
        ctx = BlockContext(ir=ir, library=library, template_id=t, confirmed=confirmed, inputs=inputs, shared=shared)
        blocks: list[Block] = [*companions, *(cls() for cls in RF_BLOCKS), BenchRfInBlock(), BenchIfOutBlock()]
        results: list[BlockResult] = []
        try:
            for blk in blocks:
                prefix = PREFIXES.get(blk.id)
                if prefix is None:
                    raise TemplateRefusal(f"block {blk.id!r} has no reference / net prefix in PREFIXES")
                if blk.id not in REGIONS:
                    raise TemplateRefusal(f"block {blk.id!r} has no floorplan region in REGIONS")
                r = blk.build(ctx, prefix)
                ctx.shared.update(r.params)
                results.append(r)
            board = merge_results(results, block_id=t)
        except TemplateRefusal as e:
            return _refused(plan, str(e))
        components, report = exclude_floating(board.components, board.nets, board.stimuli, t)
        if report.unresolved:
            return _refused(plan, "; ".join(f.reason for f in report.unresolved))
        region_choices: list[tuple[Choice, Traced]] = []
        rf_blocks: list[RFBlock] = []
        for r in results:
            vals = {}
            for axis, value in zip(("x", "y", "w", "h"), REGIONS[r.block_id]):
                key = f"floor.{r.block_id}.{axis}"
                desc = f"floorplan region of block {r.block_id}: {({'x': 'left edge', 'y': 'top edge', 'w': 'width', 'h': 'height'})[axis]} (mm, board frame)"
                tr = Traced(value=value, unit="mm", provenance=choice_provenance(t, f"{key} = {value!r} mm: {desc}", confirmed))
                region_choices.append((Choice(key, desc, value, "mm"), tr))
                vals[axis] = tr
            rf_blocks.append(RFBlock(id=r.block_id, title=r.title, refs=[c.ref for c in r.components], chain=r.chain, shield_ref=r.shield_ref,
                                     region=RFRegion(**vals), ports=r.ports))
        try:
            rf = RFDesign(blocks=rf_blocks, networks=board.networks, frequency_plan=board.plan_lines, lab_items=board.lab_items, rails=board.rails,
                          model_values=sorted(board.model_keys), profile_keys=profile_keys())
        except ValueError as e:
            return _refused(plan, f"the RF design refuses the composition: {e}")
        params: dict[str, Traced] = {c.key: tr for c, tr in profile}
        clash = sorted(k for k in board.params if k in params)
        if clash:
            return _refused(plan, f"block parameters {clash} clash with the profile's")
        params.update(board.params)
        plan.choices = [c for c, _ in profile] + board.choices + [c for c, _ in region_choices]
        plan.computed = list(board.computed)
        plan.parts = [_part_line(board.placed[c.ref]) for c in components if c.ref in board.placed]
        plan.nets = [f"{n.name} ({n.kind.value}): {', '.join(f'{p.component_ref}.{p.pin_number}' for p in n.pins)}" for n in board.nets]
        plan.simulation = self._simulation_lines(board, components)
        plan.notes.extend(board.notes)
        topology = Topology(
            name=self.title, domains=[CircuitDomain.RF, CircuitDomain.ANALOG, CircuitDomain.POWER],
            rationale=("RX_RF (U.FL) -> 2-pole top-C BPF -> BFR92 LNA -> 3-pole top-C BPF -> ADEX-10 (LO1 = f_c - IF1, low side) -> diplexer -> BFR92 IF1 "
                       "post-amp -> IF1 (U.FL); LO1 = TCXO f_R = LO1 / 12 -> x3 -> x2 (double-tuned top-C tanks) -> x2 into a 4-resonator LO BPF (the last tank and "
                       "the LO BPF as one network) -> PHA-1 -> pi pad. "
                       "The ICs and the transistors' RF behaviour have no models: their passive networks are the RF fixtures, judged under confirmed model values"),
            provenance=ctx.provenance(f"selected by radio_build = {RADIO_BUILD}"),
            blocks=[TopologyBlock(id=r.block_id, function=r.title, domain=CircuitDomain.RF if r.block_id in (FRONTEND_ID, MIXER_ID, CHAIN_ID, BUFFER_ID) else CircuitDomain.POWER,
                                  component_refs=[c.ref for c in r.components], input_nets=[p.net for p in r.ports if p.direction == "in"],
                                  output_nets=[p.net for p in r.ports if p.direction == "out"], provenance=ctx.provenance(f"block {r.block_id}"))
                    for r in results],
        )
        changes = [DesignChange(description=f"topology: {self.title}", target="topology", operation="set", payload=topology, rationale=f"template {t} v{self.version}")]
        changes += [DesignChange(description=f"add {c.ref} ({c.description})", target="components", operation="append", payload=c) for c in components]
        changes += [DesignChange(description=f"add net {n.name}", target="nets", operation="append", payload=n) for n in board.nets]
        changes += [DesignChange(description=f"parameter {k}", target=f"parameters.{k}", operation="set", payload=v) for k, v in params.items()]
        changes += [DesignChange(description=f"parameter {c.key}", target=f"parameters.{c.key}", operation="set", payload=tr) for c, tr in region_choices]
        if board.stimuli or board.analyses or board.expectations:
            changes.append(DesignChange(description="simulation setup", target="simulation", operation="set",
                                        payload=SimulationSetup(stimuli=board.stimuli, analyses=board.analyses, expectations=board.expectations)))
        changes += [DesignChange(description=f"constraint {c.id}", target="constraints", operation="append", payload=c) for c in board.constraints]
        changes.append(DesignChange(description="RF design: blocks, fixture networks, frequency plan, lab items", target="rf", operation="set", payload=rf,
                                    rationale=f"template {t}: {len(rf.networks)} fixture network(s), {len(rf.frequency_plan)} plan row(s), {len(rf.lab_items)} lab item(s)"))
        plan.changes = changes
        return plan

    @staticmethod
    def _simulation_lines(board: BlockResult, components) -> list[str]:
        lines = []
        for nw in board.networks:
            rows = ", ".join(f"{e.id} ({e.quantity} at {float(e.at.value):.9g} Hz: "
                             + (f"{e.bound} {float(e.nominal.value):.6g}" if e.bound else f"{float(e.nominal.value):.6g} +/- {float(e.tol_abs.value):.6g}") + ")"  # type: ignore[union-attr]
                             for e in nw.expectations)
            lines.append(f"fixture {nw.id}: {len(nw.members)} member(s) between {', '.join(f'{p.name} ({p.kind})' for p in nw.ports)}; rows {rows}")
        lines.append(f"every fixture PASS is a {MODEL_VERDICT}, schematic level (no track, via or ground-return inductance)")
        for e in board.expectations:
            if e.tol_rel is not None:  # the RF blocks' bias rows
                lines.append(f"design deck {e.analysis_id}: {e.id} {e.vector} = {float(e.nominal.value):.6g} +/- {float(e.tol_rel.value):.0%} under model.npn "
                             "(a bias principle under a generic transistor card, never the RF gain)")
            else:  # the power companion's rows (a bound or an absolute tolerance)
                rule = f"{e.bound} {float(e.nominal.value):.6g}" if e.bound else f"= {float(e.nominal.value):.6g} +/- {float(e.tol_abs.value):.6g}"  # type: ignore[union-attr]
                lines.append(f"design deck {e.analysis_id}: {e.id} {e.vector} {rule} ({e.provenance.note})")
        excluded = [c.ref for c in components if c.spice is not None and c.spice.exclude]
        lines.append(f"excluded from the design deck ({len(excluded)}): {', '.join(excluded)} (reasons in each part's SPICE binding)")
        return lines

    # ------------------------------------------------------------------ board

    def si_declarations(self, ir: CircuitIR, ctx: BoardContext) -> SIDeclarations:
        """``RF50`` (the 50 ohm RF / LO lines at the carrier), ``IF50`` (the mixer's IF port and IF1), ``RF_TANK_LO`` / ``RF_TANK_HI`` (high-impedance resonator nodes)."""
        from ai_eda.design.board import SIDeclarations

        out = SIDeclarations()
        z0, f_c, if1, f2 = (ctx.params.get(k) for k in ("rf.z0", "rf.f_c", "rf.if1", "lo.f2"))
        if z0 is None or f_c is None or if1 is None or f2 is None:
            return out
        tol = ctx.choice("si.rf50_z0_tol", 0.10, None, "tolerance of the RF50 / IF50 classes' impedance (10 %)")
        out.classes.append(NetClass(
            name="RF50", nets=list(RF50_NETS), target_z0_ohm=z0, z0_tol_rel=tol, rf_frequency_hz=f_c,
            description="the 50 ohm RF and LO lines (RF input, LNA ports, mixer RF and LO ports, the LO buffer input and output, the pad input) over the In1.Cu ground plane",
            provenance=ctx.structural("50 ohm RF / LO lines at the carrier"),
        ))
        out.classes.append(NetClass(
            name="IF50", nets=list(IF50_NETS), target_z0_ohm=z0, z0_tol_rel=tol, rf_frequency_hz=if1,
            description="the mixer's IF port, the post-amp's input and collector and the IF1 output: 50 ohm lines at IF1 (electrically short)",
            provenance=ctx.structural("50 ohm IF lines"),
        ))
        out.classes.append(NetClass(
            name="IF_NODE", nets=list(IF_NODE_NETS), rf_frequency_hz=if1,
            description="the diplexer's internal nodes (series L-C midpoint, absorptive branch): lumped nodes at IF1, no impedance target",
            provenance=ctx.structural("diplexer nodes"),
        ))
        out.classes.append(NetClass(
            name="RF_TANK_LO", nets=list(TANK_LO_NETS), rf_frequency_hz=f2,
            description=("the TCXO output and the first two multiplier stages' high-impedance nodes (35.5 / 106.5 / 213.1 MHz): lumped nodes, no impedance target - they must stay short "
                         "against lambda_g / 10"),
            provenance=ctx.structural("multiplier tank nodes below 300 MHz"),
        ))
        out.classes.append(NetClass(
            name="RF_TANK_HI", nets=list(TANK_HI_NETS), rf_frequency_hz=f_c,
            description=("the UHF resonator nodes (the x12 collector, the LO band-pass, both front-end band-passes): lumped nodes, no impedance "
                         "target - they must stay short against lambda_g / 10 (about 37-39 mm at 426-448 MHz)"),
            provenance=ctx.structural("UHF resonator nodes"),
        ))
        out.lines.append("RF50 / IF50 / RF_TANK_*: si.rf_length needs ir.si.rf_length_fraction, which the board module does not write yet (wave-1 gap)")
        return out

    # ------------------------------------------------------------------ report views

    def theory(self, ir: CircuitIR) -> list[TheorySection]:
        p = lambda k: parameter_value(ir, k)  # noqa: E731
        f_c, if1, lo1, image, half = p("rf.f_c"), p("rf.if1"), p("rf.lo1"), p("fe.image"), p("fe.half_if")
        n_mult, f_r, f1, f2, f3 = p("rf.n_mult"), p("lo.f_ref"), p("lo.f1"), p("lo.f2"), p("lo.f3")
        raster, bk, bf = p("kr447.channel_raster"), p("lo.birdie.k"), p("lo.birdie.f")
        spurs = {k: p(f"lo.{k}") for k in ("lo_spur_p1m", "lo_spur_p2m", "lo_spur_m1p", "lo_spur_p1p")}
        q_uhf, q_vhf, q_if1 = p("model.l_q.uhf"), p("model.l_q.vhf"), p("model.l_q.if1")
        z0, lna_r, adex_r, pha_r = p("rf.z0"), p("model.lna.port_r"), p("model.adex10.port_r"), p("model.pha1.port_r")
        rail, v_be = p("power.rx_5v"), p("fe.v_be")
        stab, err = p("rf.frequency_tolerance") or p("rf.tcxo_stability"), p("lo.lo1_error")

        def filt(name: str, key: str) -> str:
            n, bw, l = p(f"{key}.n"), p(f"{key}.bw"), p(f"{key}.l")
            order = int(n) if n is not None else 0
            couples = ", ".join(quantity(p(f"{name}.c_couple.{i}"), "F") for i in range(1, order)) or "기록 없음"
            shunts = ", ".join(quantity(p(f"{name}.c_shunt.{i}"), "F") for i in range(1, order + 1)) or "기록 없음"
            return (f"| {name} | {number(n, 2)} | {quantity(bw, 'Hz')} | {quantity(l, 'H')} | {quantity(p(f'{name}.c_tap_in'), 'F')} / {quantity(p(f'{name}.c_tap_out'), 'F')} "
                    f"| {couples} | {shunts} | {number(p(f'{name}.s21'), 4)} dB | {number(p(f'{name}.loss'), 4)} dB |")

        tank_rows = "\n".join(
            f"| lo_tank{k} | {quantity(p(f'lo.f{k}'), 'Hz')} | {quantity(p(f'lo.tank{k}.l'), 'H')} | {quantity(p(f'lo.tank{k}.bw'), 'Hz')} "
            f"| {number(_rp(p('lo.tank_qe'), p(f'lo.f{k}'), p(f'lo.tank{k}.l')), 5)} Ω | {number(p(f'lo_tank{k}.s21'), 4)} dB "
            f"| {quantity(p(f'lo_tank{k}.port_r'), 'ohm')} + j{quantity(p(f'lo_tank{k}.port_x'), 'ohm')} / {quantity(p(f'lo.tank{k}.r_load_eff'), 'ohm')} "
            f"| {number(p(f'lo_tank{k}.rel_m'), 4)} / {number(p(f'lo_tank{k}.rel_p'), 4)} dB |" for k in (1, 2))
        bias_rows = "\n".join(
            f"| {label} | {quantity(p(f'{key}.r_b1'), 'ohm')} / {quantity(p(f'{key}.r_b2'), 'ohm')} | {number(p(f'{key}.v_b'), 4)} V | {quantity(p(f'{key}.r_e'), 'ohm')} "
            f"| {quantity(p(f'{key}.ic'), 'A')} |" for label, key in (("LNA Q601", "fe.lna"), ("IF1 후치 증폭기 Q650", "fe.ifa"), ("x3 Q701", "lo.x3"),
                                                                        ("x2 Q702", "lo.x6"), ("x2 Q703", "lo.x12")))
        nf = {k: p(f"fe.nf.{k}") for k in ("after_mixer", "at_mixer", "at_bpf3", "at_lna", "total")}
        return [
            TheorySection("개요: 수신 프런트엔드 시험 보드(3단계)", (
                "신호 흐름: RF 입력(U.FL, RX_RF) → 2단 상단 결합 대역통과 필터(fe_bpf2) → BFR92 저잡음 증폭기(LNA) → 3단 대역통과 필터(fe_bpf3) → ADEX-10 1차 혼합기 "
                "→ 다이플렉서 → BFR92 IF1 후치 증폭기 → IF1 출력(U.FL, 21.4 MHz). 1차 국부 발진(LO1)은 TCXO → x3 → x2 → x2 체배기(앞의 두 단 뒤에 복동조 탱크, "
                "마지막 단 뒤에 4공진기 LO 대역통과 필터) → PHA-1 버퍼 → π 감쇠기 → 혼합기 LO 포트로 만듭니다. 이 보드와 2단계 IF 백엔드 보드를 케이블로 이으면 수신 전용 무전기가 됩니다.\n\n"
                "전원은 1단계 오디오/PTT 보드와 같은 전원 블록의 수신 부분입니다(2S 팩 → 퓨즈 → 주 P-FET 스위치 → LP38693DT-5.0 RX_5V → LP5907 RX_3V3, 참조 1xx; "
                "송신부가 없어 RX 레일은 항상 켜짐). 모든 단의 바이어스가 이 레일 값(power.rx_5v, power.rx_3v3)에서 계산됩니다.\n\n"
                "검증 범위: ADEX-10, PHA-1, TCXO 는 SPICE 모델이 없어 모든 넷리스트에서 제외되고, BFR92 는 S-파라미터가 없어 RF 이득·잡음지수·정합을 계산하지 않습니다. "
                "대신 이 부품들 사이의 수동 회로망(필터, 탱크, 다이플렉서, 감쇠기)을 RF 픽스처로 잘라 ngspice 교류 해석으로 판정합니다. 그 PASS 는 '확인된 모델값 아래의 "
                "회로망 판정(측정된 부품이 아님)'이며 트랙·비아·접지 귀환 인덕턴스가 없는 회로도 수준입니다. 설계 덱은 다섯 트랜지스터의 바이어스 전류만 일반 NPN 카드"
                "(model.npn)로 확인합니다. 인덕터 Q, 포트 저항, 잡음지수 등 model.* 값과 KR 447 MHz 규제 수치(kr447.*)는 모두 검증되지 않은(UNVERIFIED) 선택값입니다. "
                "이 보드는 수신 전용이지만, 무전기로 송신하려면 KC 적합성평가가 먼저 필요합니다."
            )),
            TheorySection("주파수 계획", (
                f"- 수신 채널 f_c = {quantity(f_c, 'Hz')} (요구사항; 검증되지 않은 KR 447 MHz 채널 격자 {quantity(raster, 'Hz')} 위의 채널이어야 함).\n"
                f"- IF1 = {quantity(if1, 'Hz')}, 저측 LO: LO1 = f_c − IF1 = {quantity(lo1, 'Hz')} (`calc.rf.superhet.lo`). 고측 LO 였다면 영상이 490.36 MHz(UHF TV 대역, 검증되지 않음)에 옵니다.\n"
                f"- 영상 주파수 = 2·LO1 − f_c = {quantity(image, 'Hz')} (`calc.rf.superhet.image`), 반-IF(2x2) 응답 = (f_c + LO1)/2 = {quantity(half, 'Hz')} (`calc.rf.superhet.half_if`).\n"
                f"- 체배 계수 N = {number(n_mult, 3)} = 3 × 2 × 2 (결정 1B), 기준 f_R = LO1 / N = {quantity(f_r, 'Hz', 9)} (`calc.clock.divided`), 단 출력 "
                f"{quantity(f1, 'Hz')} / {quantity(f2, 'Hz')} / {quantity(f3, 'Hz')} (`calc.rf.mult.stage`).\n"
                f"- LO 스퓨리어스 응답 LO1 + k·f_R ± IF1 (`calc.rf.superhet.lo_spur_response`): {quantity(spurs['lo_spur_p1m'], 'Hz')} (k = +1, −IF1: 영상 + f_R, 반송파에서 가장 가까움), "
                f"{quantity(spurs['lo_spur_p2m'], 'Hz')} (k = +2, −IF1), {quantity(spurs['lo_spur_m1p'], 'Hz')} (k = −1, +IF1), {quantity(spurs['lo_spur_p1p'], 'Hz')} (k = +1, +IF1). "
                "이 행들은 자체 판정이 없고, 탱크·LO 필터의 픽스처 행과 실험실 항목(rf.lab.lo_spur_response)을 가리킵니다.\n"
                f"- 버디(birdie) 여유: f_R 의 {number(bk, 3)}차 고조파 {quantity(bf, 'Hz')} 가 f_c 에서, f_R 자체가 IF1 에서 채널 격자 {quantity(raster, 'Hz')} 이상 떨어져야 합니다 "
                "(rf.freq_plan 의 margin 행, 산술).\n"
                f"- 주파수 오차: TCXO 안정도 {number(stab, 3)} ppm 은 LO1 에서 {quantity(err, 'Hz')} (`calc.rf.ppm_offset`; 체배해도 ppm 은 유지)."
            )),
            TheorySection("프런트엔드·LO 대역통과 필터(상단 결합, Dishal)", (
                "공진기 n 개를 상단 결합 커패시터로 잇고 양 끝을 용량성 탭으로 포트에 연결한 버터워스 회로망입니다(Dishal 1949; Zverev 1967):\n\n"
                "    Q_e = g_1 f0 / BW,  R_p = Q_e ω0 L,  C_res = 1/(ω0² L)\n"
                "    C_s = 1/(ω0 sqrt(R_p R_t − R_t²))   (탭; R_t < R_p 이어야 함 - 탭은 낮추는 방향으로만 변환)\n"
                "    C_k = k_ij (BW/f0) C_res,  k_ij = 1/sqrt(g_i g_j),  C_i = C_res − 인접 결합 − 탭의 병렬 등가\n\n"
                f"포트: RX_RF {quantity(z0, 'ohm')}, LNA {quantity(lna_r, 'ohm')} (model.lna.port_r), 혼합기 {quantity(adex_r, 'ohm')} (model.adex10.port_r), PHA-1 {quantity(pha_r, 'ohm')}; "
                f"인덕터 손실은 직렬 R = ω0 L / Q_u, Q_u = {number(q_uhf, 3)} (model.l_q.uhf).\n\n"
                "| 회로망 | 공진기 | BW | L | 탭(입력/출력) | 결합 C | 공진 C | S21(f0) | Cohn 손실 |\n|---|---|---|---|---|---|---|---|---|\n"
                f"{filt('fe_bpf2', 'fe.bpf2')}\n{filt('fe_bpf3', 'fe.bpf3')}\n{filt('lo_bpf', 'lo.bpf')}\n\n"
                f"거부(f0 기준): fe_bpf2 영상 {number(p('fe_bpf2.rel_image'), 4)} dB; fe_bpf3 영상 {number(p('fe_bpf3.rel_image'), 4)} dB, LO1 {number(p('fe_bpf3.rel_lo1'), 4)} dB; "
                f"lo_bpf LO1 ∓ f_R {number(p('lo_bpf.rel_m1'), 4)} / {number(p('lo_bpf.rel_p1'), 4)} dB, ∓ 2 f_R {number(p('lo_bpf.rel_m2'), 4)} / {number(p('lo_bpf.rel_p2'), 4)} dB.\n\n"
                "상단 결합 회로망은 이 오프셋에서 비대칭이라(위쪽이 약함) 대칭 협대역 식(10 log10(1 + x⁴/4) 등)은 천장값일 뿐이고, 픽스처 공칭값은 정확한 회로망 응답"
                "(`calc.rf.resonator.top_c.s21_db` / `.rel_s21_db`)입니다. 허용오차 1 dB 는 컴파일된 넷리스트가 설계한 회로망을 구현했는지를 봅니다. "
                "설계 문서의 단측 한계(예: fe_bpf2 ≥ −4 dB)는 같은 모델에서 1 dB 를 뺀 값이므로 같은 회로망을 판정합니다. Cohn 손실 L0 = 4.343 Σg_i f0/(BW Q_u) "
                "(`calc.rf.bpf.dissipation_loss`)는 감도 예산에 씁니다(회로망보다 약간 보수적)."
            )),
            TheorySection("LNA·후치 증폭기·체배기의 바이어스", (
                f"모든 BFR92 단은 분압 바이어스입니다: V_B = V_rail·R2/(R1 + R2) (`calc.divider.v_out`, V_rail = RX_5V {quantity(rail, 'V')}), "
                f"I_C ≈ (V_B − V_BE)/R_E (`calc.rf.bjt_bias.ic`, 베이스 전류 무시, V_BE = {quantity(v_be, 'V')} 선택값).\n\n"
                "| 단 | R1 / R2 | V_B | R_E | I_C(공칭) |\n|---|---|---|---|---|\n" + bias_rows + "\n\n"
                "각 단의 컬렉터 전원은 0 Ω 링크를 거칩니다. 설계 덱은 이 링크를 0 V 전원(이상적 전류계)으로 시뮬레이션해 I_C 를 읽고, 공칭값 ±15 % 로 판정합니다 "
                "(베이스 전류와 V_BE 변화 때문에 공식보다 약간 작게 나옴). 실험실에서는 링크를 떼고 전류계를 넣습니다. 이것은 일반 NPN 카드 아래의 바이어스 원리 확인일 뿐, "
                "LNA 이득·잡음지수나 체배기의 고조파 발생은 모델이 없습니다. 체배기는 A급 바이어스에서 과구동되어 고조파를 만든다고 가정하며(검증되지 않음), 그 변환 이득은 실험실 항목입니다."
            )),
            TheorySection("혼합기와 다이플렉서", (
                f"ADEX-10(수동 이중 평형 혼합기, +7 dBm LO; 검증되지 않음)의 IF 포트는 IF1 뿐 아니라 LO1·RF·합성분 {quantity(p('fe.sum_product'), 'Hz')} (LO1 + f_c)도 내보냅니다. "
                "다이플렉서는 IF1 을 후치 증폭기로 통과시키고 나머지를 50 Ω 로 흡수해 혼합기가 광대역으로 종단되게 합니다:\n\n"
                f"- 직렬 L–C 가 IF1 에서 공진: L = {quantity(p('fe.dip.l'), 'H')}, C = 1/(ω_IF1² L) = {quantity(p('fe.dip.c_series'), 'F')} (`calc.rf.lc.c_for_resonance`), "
                f"인덕터 Q {number(q_if1, 3)} (model.l_q.if1).\n"
                f"- 흡수 가지: C = {quantity(p('fe.dip.c_hp'), 'F')} + R = {quantity(p('fe.dip.r_term'), 'ohm')} 접지.\n\n"
                f"판정: IF1 에서 S21 ≥ {number(p('fe.dip.s21_min'), 3)} dB, LO1 과 합성분에서 혼합기 쪽 S11 ≤ {number(p('fe.dip.s11_max'), 3)} dB (단측 한계; 선택값)."
            )),
            TheorySection("LO 체인: 체배기 탱크와 버퍼", (
                f"x3, x2 단 뒤의 탱크는 공진기 2개의 상단 결합 회로망(복동조)이며, 끝단 외부 Q 를 Q_e = {number(p('lo.tank_qe'), 3)} 으로 정해 BW = g_1 f0 / Q_e "
                "(`calc.rf.resonator.top_c.bw_for_qe`)입니다. 포트는 설계 문서의 BFR92 포트 모델(컬렉터 "
                f"{quantity(p('model.bfr92.r_out'), 'ohm')} model.bfr92.r_out, 다음 베이스 {quantity(p('model.bfr92.r_in'), 'ohm')} model.bfr92.r_in)이지만, "
                "컬렉터 노드에는 급전 초크가 0 Ω 링크와 디커플링을 거쳐 AC 접지로 병렬로 붙어 있고(그 리액턴스가 1 kΩ 포트와 같은 크기), 다음 베이스 노드에는 "
                "그 단의 분압기가 붙어 있습니다. 그래서 입력 탭은 Z = R_port ∥ (jω0 L_ch + ω0 L_ch / Q) 의 실수부 R' 를 변환하고 허수부 X' 를 흡수하며"
                "(C_s = 1/(ω0 (X' + sqrt(R' (R_p − R')))), `calc.rf.resonator.top_c.port_r` / `.port_x` / `.c_tap_reactive`), 부하 쪽은 500 Ω ∥ 분압기"
                "(`lo.tank<k>.r_load_eff`)로 설계합니다. 초크와 0 Ω 링크, 다음 단의 분압 저항은 픽스처의 구성원이고(레일은 RX_5V 의 이상 전원), 공칭값은 그 "
                "회로망의 정확한 응답(`calc.rf.resonator.top_c.ported_s21_db` / `.ported_rel_s21_db`)입니다.\n\n"
                f"마지막 x2 단은 탱크와 LO 대역통과 필터를 따로 두지 않고 {number(p('lo.bpf.n'), 2)}공진기 필터 하나(lo_bpf)로 콜렉터에서 PHA-1 입력 "
                f"{quantity(p('model.pha1.port_r'), 'ohm')} (model.pha1.port_r)까지 잇습니다. 설계 문서는 마지막 탱크와 2단 LO 필터를 50 Ω 중간 임피던스로 잇고 두 "
                "픽스처를 따로 판정했지만, 실제 회로에서 두 필터는 탭 커패시터로 직접 이어지고 그 사이에 저항성 노드가 없어 두 판정의 합이 실제 연결을 나타내지 "
                "않았습니다.\n\n"
                "탭은 낮추는 방향으로만 변환하므로 R_p = Q_e ω0 L 이 초크와 병렬인 컬렉터 포트(R')보다 커야 합니다. 설계 문서의 50 Ω 예시 값(300 MHz 아래 30 pF, 위 4.7 nH)은 "
                "R_p 948 / 474 / 264 Ω 이 되어 계산기가 거부하므로, 여유 있게 큰 L 을 골랐습니다(비대칭이 몇 dB 달라지지만 공칭값이 자기 포트에서 계산되므로 일관됨).\n\n"
                "| 탱크 | f0 | L | BW | R_p | S21(f0) | 원천 R' + jX' / 부하 | f0 ∓ f_R (f0 기준) |\n|---|---|---|---|---|---|---|---|\n" + tank_rows + "\n\n"
                f"Q_u = {number(q_vhf, 3)} (300 MHz 아래, model.l_q.vhf) / {number(q_uhf, 3)} (위; lo_bpf). 트리머는 넣지 않았습니다: Murata TZB4-A 발자국(7.5 × 4.5 mm) 6개가 "
                "BMI-S-103 차폐 캔 안쪽 면적의 절반을 넘습니다. 정렬은 부품 선별로 하며 실험실 항목입니다.\n\n"
                f"PHA-1 버퍼 뒤의 정합 π 감쇠기 {number(p('lo.pad.a_db'), 3)} dB: R_sh = Z0 (K + 1)/(K − 1) = {quantity(p('lo.pad.r_shunt'), 'ohm')}, "
                f"R_se = Z0 (K − 1/K)/2 = {quantity(p('lo.pad.r_series'), 'ohm')}, K = 10^(A/20) (`calc.rf.attenuator.pi.*`). 픽스처는 S21 = "
                f"{number(p('lo_pad.s21'), 4)} dB ± 0.2 dB 와 S11 ≤ −20 dB 를 판정합니다. 혼합기의 +7 dBm LO 레벨은 실험실 항목입니다."
            )),
            TheorySection("감도 예산(추정, 판정 아님)", (
                "뒤에서부터 Friis 식 F = F_1 + (F_rest − 1)/G_1 (`calc.rf.friis_nf`)로 합치고, 수동 손실 L 은 앞에 두면 dB 로 더해집니다(F = L·F_rest, `calc.rf.db_sum`):\n\n"
                f"- IF 백엔드 NF {number(p('model.ifb.nf'), 3)} dB, 후치 증폭기 NF {number(p('model.ifamp.nf'), 3)} dB / 이득 {number(p('model.ifamp.gain'), 3)} dB → {number(nf['after_mixer'], 4)} dB\n"
                f"- 혼합기 변환 손실 {number(p('model.adex10.cl'), 3)} dB → {number(nf['at_mixer'], 4)} dB; fe_bpf3 손실 {number(p('fe_bpf3.loss'), 4)} dB → {number(nf['at_bpf3'], 4)} dB\n"
                f"- LNA NF {number(p('model.lna.nf'), 3)} dB / 이득 {number(p('model.lna.gain'), 3)} dB → {number(nf['at_lna'], 4)} dB; fe_bpf2 손실 {number(p('fe_bpf2.loss'), 4)} dB → "
                f"**NF ≈ {number(nf['total'], 4)} dB**\n"
                f"- 감도 = kTB + NF + C/N (`calc.rf.sensitivity`, T0 {number(p('fe.t0'), 4)} K, B = {quantity(p('rf.if_bw'), 'Hz')}, C/N {number(p('fe.snr'), 3)} dB) "
                f"≈ **{number(p('fe.sensitivity'), 4)} dBm**\n\n"
                "모든 잡음지수·이득·변환 손실은 검증되지 않은 모델값입니다. 무전기 전체(결정 2A)는 LPF 와 T/R 스위치 손실(약 1.3 dB)이 더해져 약 −113 dBm 이 v1 추정값입니다. "
                "실제 12 dB SINAD 감도는 실험실 항목(rf.lab.sensitivity)이며, 요구사항 rx_sensitivity 가 있으면 그 항목에 추정값과 함께 기록될 뿐 PASS 도 거부도 되지 않습니다."
            )),
        ]

    def theory_figures(self, ir: CircuitIR) -> list[Figure]:
        """The front-end filters' exact S21 (each and in cascade) around the carrier; the LO chain's tanks and band-pass around their stage frequencies."""
        from ai_eda.design.templates import THEORY_CURVE_NOTE, _curve_figure

        p = lambda k: parameter_value(ir, k)  # noqa: E731
        out: list[Figure] = []
        f_c, z0, lna_r, adex_r, q = p("rf.f_c"), p("rf.z0"), p("model.lna.port_r"), p("model.adex10.port_r"), p("model.l_q.uhf")
        f2 = [p(k) for k in ("fe.bpf2.n", "fe.bpf2.bw", "fe.bpf2.l")]
        f3 = [p(k) for k in ("fe.bpf3.n", "fe.bpf3.bw", "fe.bpf3.l")]
        image, lo1, half, near = p("fe.image"), p("rf.lo1"), p("fe.half_if"), p("lo.lo_spur_p1m")
        if None not in (f_c, z0, lna_r, adex_r, q, image, lo1, half, *f2, *f3):
            try:
                xs = [350e6 + 200e6 * i / 400 for i in range(401)]
                y2 = [radio.top_c_s21_db_value(f2[0], f_c, f2[1], f2[2], z0, lna_r, q, x) for x in xs]
                y3 = [radio.top_c_s21_db_value(f3[0], f_c, f3[1], f3[2], lna_r, adex_r, q, x) for x in xs]
            except ValueError:
                xs = []
            if xs:
                bands = [("x", image, image, "영상"), ("x", lo1, lo1, "LO1"), ("x", half, half, "반-IF"), ("x", f_c, f_c, "f_c")]
                if near is not None:
                    bands.append(("x", near, near, "영상 + f_R"))
                caption = (f"'프런트엔드·LO 대역통과 필터' 절의 fe_bpf2 · fe_bpf3 정확한 S21 (`calc.rf.resonator.top_c.s21_db`, Q_u {number(q, 3)}, 각자의 포트 저항) 과 "
                           "두 필터의 dB 합(LNA 이득 제외). 안내선 = 영상, LO1, 반-IF, 영상 + f_R, 반송파. " + THEORY_CURVE_NOTE)
                out.append(_curve_figure("theory_fe_s21", "프런트엔드 필터 S21", caption,
                                         [("fe_bpf2", xs, y2), ("fe_bpf3", xs, y3), ("합", xs, [a + b for a, b in zip(y2, y3)])],
                                         x_label="주파수 (Hz)", y_label="S21 (dB)", bands=bands))
        r_out, pha_r, qe = p("model.bfr92.r_out"), p("model.pha1.port_r"), p("lo.tank_qe")
        series: list[tuple[str, list[float], list[float]]] = []
        guides: list[tuple[str, float, float, str]] = []
        for k, stage in ((1, "x3"), (2, "x6")):
            fk, l, bw, qk = p(f"lo.f{k}"), p(f"lo.tank{k}.l"), p(f"lo.tank{k}.bw"), p("model.l_q.vhf")
            l_ch, r_in, r_eff = p(f"lo.{stage}.l_choke"), p("model.bfr92.r_in"), p(f"lo.tank{k}.r_load_eff")
            if None in (fk, l, bw, qk, r_out, l_ch, r_in, r_eff):
                continue
            try:
                xs = [fk * (0.6 + 0.8 * i / 200) for i in range(201)]
                series.append((f"lo_tank{k}", xs, [radio.top_c_ported_s21_db_value(2, fk, bw, l, r_out, l_ch, qk, r_in, r_eff, qk, x) for x in xs]))
                guides.append(("x", fk, fk, f"f{k}"))
            except ValueError:
                pass
        bpf = [p(k) for k in ("lo.bpf.n", "lo.bpf.bw", "lo.bpf.l", "lo.f3", "lo.x12.l_choke")]
        if None not in (*bpf, r_out, pha_r, q):
            try:
                xs = [bpf[3] * (0.6 + 0.8 * i / 200) for i in range(201)]
                series.append(("lo_bpf", xs, [radio.top_c_ported_s21_db_value(bpf[0], bpf[3], bpf[1], bpf[2], r_out, bpf[4], q, pha_r, pha_r, q, x) for x in xs]))
                guides.append(("x", bpf[3], bpf[3], "LO1"))
            except ValueError:
                pass
        if series:
            caption = (f"'LO 체인' 절의 두 복동조 탱크(Q_e {number(qe, 3)})와 LO 대역통과 필터의 정확한 S21 (`calc.rf.resonator.top_c.ported_s21_db`: 컬렉터 모델 저항 ∥ 급전 초크 → "
                       "다음 베이스 ∥ 분압기 / PHA-1). "
                       "가로축은 로그 눈금. " + THEORY_CURVE_NOTE)
            out.append(_curve_figure("theory_lo_s21", "LO 체인 탱크·필터 S21", caption, series, x_label="주파수 (Hz)", y_label="S21 (dB)", log_x=True, bands=guides))
        return out

    def part_notes(self, ir: CircuitIR) -> dict[str, PartNote]:
        from ai_eda.design.rf.t_audio_ptt import stage1_part_notes, stage1_refs

        p = lambda k: parameter_value(ir, k)  # noqa: E731
        stage1 = stage1_refs(ir)  # the P9 power companion's parts: the stage-1 board's notes
        out: dict[str, PartNote] = stage1_part_notes(ir, stage1)
        q_uhf, q_vhf = p("model.l_q.uhf"), p("model.l_q.vhf")
        for c in ir.components:
            if c.ref in stage1:
                continue
            d = c.description
            v = f"값 {c.value}"
            if d.startswith(("fe_bpf2", "fe_bpf3", "lo_bpf", "lo_tank")):
                net = d.split(" ", 1)[0]
                ind = "inductor" in d
                out[c.ref] = PartNote(
                    f"{net} 회로망의 {'공진 인덕터' if ind else '커패시터(탭·결합·공진)'}", "`calc.rf.resonator.top_c.*` 가 계산한 값 그대로(E 계열 반올림 없음; 픽스처 공칭값과 같은 회로망)",
                    [v, (f"Q ≥ {number(q_uhf if 'tank' not in net else q_vhf, 3)} (모델값) @ 동작 주파수, 자기공진 주파수가 동작 주파수의 2배 이상"
                         if ind else "C0G/NP0, ±0.1 pF 또는 ±1 % (1 pF 이하 값은 패드 기생 용량과 같은 크기 - 실장 후 정렬)")],
                    [unverified("Coilcraft 0604HQ / 0603CS" if ind else "Murata GJM 0402 C0G")])
            elif "LNA Q1" in d and "BFR92" in d and ":" not in d:
                out[c.ref] = PartNote("저잡음 증폭기(A급)", "광대역 NPN, 라이브러리 설명 '5GHz'; S-파라미터가 없어 정합은 설계하지 않고 50 Ω 포트 모델로 둠",
                                      [f"I_C 공칭 {quantity(p('fe.lna.ic'), 'A')}", "NF, 이득은 검증되지 않은 모델값(2 dB / 15 dB)"],
                                      [unverified("NXP BFR92AW"), unverified("Infineon BFR92P")])
            elif "post-amp Q50" in d and "BFR92" in d and ":" not in d:
                out[c.ref] = PartNote("IF1 후치 증폭기(A급)", "혼합기 변환 손실을 보상; IF 백엔드 입력 50 Ω 을 구동", [f"I_C 공칭 {quantity(p('fe.ifa.ic'), 'A')}"],
                                      [unverified("NXP BFR92AW")])
            elif "LO multiplier stage" in d and "BFR92" in d and ":" not in d:
                out[c.ref] = PartNote("LO 체배기(과구동 A급, 고조파 발생)", "고조파 발생은 모델링하지 않음; 바이어스만 설계 덱에서 확인",
                                      ["fT 가 출력 주파수의 10배 이상", "변환 이득과 스퓨리어스는 실험실 항목"], [unverified("NXP BFR92AW")])
            elif "base-divider" in d or "emitter resistor" in d:
                out[c.ref] = PartNote("바이어스 저항", "V_B = V_rail R2/(R1+R2), I_C = (V_B − V_BE)/R_E 의 선택값", [v, "±1 %"], [unverified("0402 1 % 후막 저항")])
            elif "emitter bypass" in d or "decoupling" in d or "DC block" in d or "coupling capacitor TCXO" in d:
                out[c.ref] = PartNote("바이패스·디커플링·DC 차단 커패시터", "동작 주파수에서 임피던스가 포트 저항보다 충분히 작은 값(선택값)", [v, "X7R/C0G, 자기공진 확인"],
                                      [unverified("Murata GRM 0402")])
            elif "0 ohm bias-current link" in d:
                out[c.ref] = PartNote("0 Ω 바이어스 전류 링크", "설계 덱은 0 V 전원(전류계)으로 I_C 를 읽음; 실험실에서 떼고 전류계를 넣는 측정점", ["0 Ω 점퍼, 1 A 이상"],
                                      [unverified("0402 0 Ω 점퍼")])
            elif "choke" in d and d.startswith("LO multiplier stage"):
                out[c.ref] = PartNote("체배 단 급전 초크(탱크·필터 회로망의 구성원)",
                                      "리액턴스가 1 kΩ 컬렉터 포트와 같은 크기라 회로망 설계에 들어감: 입력 탭이 그 리액턴스를 흡수하고, 픽스처가 초크(Q 모델값)를 포함해 판정",
                                      [v, "값과 Q 가 설계값에 맞을 것", "자기공진 주파수가 단 출력 주파수보다 높을 것(검증되지 않음)"],
                                      [unverified("Coilcraft 0603CS / Murata LQW18")])
            elif "choke" in d:
                out[c.ref] = PartNote("급전 초크", "동작 주파수에서 포트 저항보다 큰 리액턴스(선택값); 픽스처에는 포함되지 않음(개방으로 가정)",
                                      [v, "자기공진 주파수가 동작 주파수보다 높을 것(검증되지 않음)"], [unverified("Coilcraft 0603CS / Murata LQW18")])
            elif "first mixer" in d:
                out[c.ref] = PartNote("1차 혼합기(수동 이중 평형)", "50 Ω 포트로 다이플렉서 설계가 단순; 2x2 억압이 반-IF 응답을 담당(검증되지 않음)",
                                      ["10–1000 MHz (라이브러리 설명)", "LO +7 dBm", "변환 손실 약 7 dB (모델값)"],
                                      [unverified("Mini-Circuits ADEX-10+"), unverified("ADI LT5560", "능동, 저전류; 부품표의 대체안")])
            elif d.startswith("diplexer"):
                out[c.ref] = PartNote("IF 다이플렉서 소자", "IF1 통과(직렬 L–C 공진), 고주파 성분 50 Ω 흡수", [v], [unverified("0603 인덕터 / 0402 C0G / 0402 저항")])
            elif "reference TCXO" in d:
                out[c.ref] = PartNote("LO1 기준 TCXO", f"f_R = LO1 / N = {quantity(p('lo.f_ref'), 'Hz', 9)} 주문 주파수", [
                    f"안정도 ≤ {number(p('rf.frequency_tolerance') or p('rf.tcxo_stability'), 3)} ppm", "클리핑 사인 출력, 3.3 V",
                    "채널이 공식 문서로 확인되기 전에는 주문하지 말 것"], [unverified("Kyocera KT2520K 주문품"), unverified("NDK / Epson 2520 TCXO 주문품")])
            elif "LO1 buffer" in d:
                out[c.ref] = PartNote("LO1 버퍼 증폭기", "50 Ω 광대역 이득 블록(라이브러리 설명 50–6000 MHz)으로 혼합기 LO 구동", ["출력 약 +13 dBm (검증되지 않음)", "RF_OUT 초크 바이어스"],
                                      [unverified("Mini-Circuits PHA-1+"), unverified("Mini-Circuits GALI 계열")])
            elif d.startswith("LO pad"):
                out[c.ref] = PartNote("LO 감쇠기(정합 π)", f"`calc.rf.attenuator.pi.*` ({number(p('lo.pad.a_db'), 3)} dB)", [v, "±1 %, 0402"], [unverified("0402 1 % 박막 저항")])
            elif "shield can" in d:
                out[c.ref] = PartNote("차폐 캔", "LNA·필터 또는 LO 체인을 서로와 외부로부터 차폐(펜스 패드는 GND)",
                                      ["펜스 안쪽에 블록 부품 전체가 들어갈 것(배치기가 확인)"], [unverified("Laird BMI-S-103 (프레임+커버)")])
            elif "U.FL" in d:
                out[c.ref] = PartNote("벤치 RF 포트", "신호 발생기 입력 / IF 백엔드 보드로의 출력(50 Ω)", ["50 Ω 동축 커넥터"], [unverified("Hirose U.FL-R-SMT-1")])
            elif "bench supply header" in d:
                out[c.ref] = PartNote("벤치 전원 헤더(RX 전원부 대체)", d, ["2.54 mm 핀 헤더"], [unverified("일반 1x3 핀 헤더")])
        return out


_REF = re.compile(r"^[A-Za-z][A-Za-z_]*[0-9]+$")


def _part_line(placed: PlacedPart) -> str:
    """One table line: the board reference, library ids, value, this board's description of the part and the pin notes.

    The parts table's own role text (``part_line``) names every use of a shared
    row (an NPN "LNA Q601, IF1 post-amplifier Q602, ..."), so the line shows
    the component's description instead; a pin note written with the
    block-local reference before re-basing names the board reference.
    """
    c = placed.component
    notes = []
    for note in placed.notes:
        head, _, rest = note.partition(" ")
        notes.append(f"{c.ref} {rest}" if _REF.match(head) and head != c.ref else note)
    tail = f"; {'; '.join(notes)}" if notes else ""
    return f"{c.ref} {placed.part.lib_id} / {placed.part.footprint_id}, value {c.value} ({c.description}){tail}"


def _rp(qe: float | None, f: float | None, l: float | None) -> float | None:
    """R_p = Q_e w0 L (the end resonator's required parallel resistance), for the theory table."""
    return None if None in (qe, f, l) else qe * 2.0 * math.pi * f * l  # type: ignore[operator]


#: the SI classes' nets (the blocks' net names; the stage board prefixes no internal net)
RF50_NETS: tuple[str, ...] = ("RX_RF", "LNA_IN", "LNA_OUT", "MIX_RF", "LO1_RAW", "LO1_BUF_OUT", "LO_PAD_IN", "LO1_MIX")
IF50_NETS: tuple[str, ...] = ("MIX_IF", "IFA2_B", "IFA2_C", "IF1")
#: the diplexer's internal nodes (lumped, no impedance target)
IF_NODE_NETS: tuple[str, ...] = ("DIP_M", "DIP_T")
TANK_LO_NETS: tuple[str, ...] = ("LO_TCXO", "LO_X3_B", "LO_X3_C", "LO_T1_R1", "LO_T1_R2", "LO_X6_B", "LO_X6_C", "LO_T2_R1", "LO_T2_R2", "LO_X12_B")
TANK_HI_NETS: tuple[str, ...] = ("LO_X12_C", "LO_B_R1", "LO_B_R2", "LO_B_R3", "LO_B_R4", "FE2_R1", "FE2_R2", "FE3_R1", "FE3_R2", "FE3_R3")


__all__ = [
    "BUILD",
    "PLANE_REASON",
    "PREFIXES",
    "RADIO_BUILD",
    "REGIONS",
    "RF_BLOCKS",
    "V_IN_RANGE",
    "BenchIfOutBlock",
    "BenchRfInBlock",
    "BenchSupplyBlock",
    "KR447RxFrontendTemplate",
    "default_companions",
]
