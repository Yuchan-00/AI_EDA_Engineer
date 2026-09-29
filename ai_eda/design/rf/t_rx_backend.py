"""The ``kr447_rx_backend`` template (``radio_build = rx_backend``): the stage-2 bench board of the KR 447 MHz FM receiver (kr447 design §2.2).

Invariant: a pure function of the confirmed requirements, the library on
disk and the block builders - nothing is guessed, nothing is grounded that is
not. The board is the composition (:func:`~ai_eda.design.rf.blocks.base.merge_results`)
of the IF back-end block (:class:`~ai_eda.design.rf.blocks.if_backend.IfBackendBlock`,
references ``5xx``) and its *companions*: the RX power section and the RX
audio block of part P9 (:func:`default_companions`:
``PowerBlock(modes=("rx",))``, references ``1xx``, internal nets ``PWR_*``,
and ``RxAudioBlock()``, references ``4xx``, internal nets ``RXA_*``; kr447
design §2.1). The power block reads ``input_voltage`` (6.4-8.4 V: above the
confirmed pack cut-off, at most a full 2S pack; no TX inhibit on a
receive-only board) and supplies ``RX_5V`` / ``RX_3V3``; the RX audio block
reads ``audio_bandwidth`` (2-4 kHz, its low-pass's design range), takes
``DISC_OUT`` / ``RSSI`` and drives ``MUTE``; its volume pot's wiper reaches
only the unmodelled LM386 and is reported, kept (the pot's own resistors give
it a DC path). The design deck is the companions' (the IF back-end's parts
are fixture members only): ``main_switch_on``, the RX audio chain's three
``ac`` rows and ``sq_threshold``, with ``DISC_OUT`` driven by the detector-level
test tone. A template built with no companions **refuses** instead of
building a board without a supply; :class:`BenchInterfaceBlock` - two bench
headers standing in for the P9 blocks (the RX_5V supply and MUTE in,
DISC_OUT and RSSI out) - is what most tests compose
(``KR447RxBackendTemplate(companions=(BenchInterfaceBlock(),))``), a wiring
that is honest about being a bench stand-in (its plan note says so, and
``input_voltage`` / ``audio_bandwidth`` are then served by nothing).

Selection and the closed world (kr447 design §2.0): the template is selected
only by the confirmed categorical requirement ``radio_build = rx_backend``
(:meth:`triggered_by`); it needs ``modulation`` (which must read ``fm``: the
KR 447 class is FM [UNVERIFIED]) and ``input_voltage`` (read by the RX power
section); it serves the keys of ``family.BUILDS["rx_backend"]`` and refuses
every other confirmed design requirement with the family's sentence naming
the builds that serve it (``carrier_frequency is served by
radio_build=rx_frontend, ...``). A stated ``channel_spacing`` other than the
profile's raster refuses (the ladder is designed for that channel), a stated
``pcb_layers`` other than 4 refuses before anything is asked (the layer
policy: the RF lines need a reference plane).

What the plan carries: every ``kr447.*`` regulatory placeholder as a choice
whose description ends ``[UNVERIFIED: ...]`` (``ir.rf.profile_keys``; never
grounded, ``rf.regulatory_profile`` never PASSes), every block choice and
``model.*`` value (``ir.rf.model_values``: UNVERIFIED, ``rf.model_grounding``
NOT_VERIFIED), the floorplan regions (``floor.<block>.x`` / ``.y`` / ``.w`` /
``.h`` mm, choices; ``placement.rf_floorplan`` packs each block into its
region and the outline is their bounding box), the RF design ``ir.rf``
(blocks, fixture networks, the frequency plan, lab items, rails) and the
signal-integrity classes (:meth:`si_declarations`: ``IF50`` - the 50 ohm IF1
input, an RF net at IF1 with the system impedance as its target - and
``IF_HIZ`` - the ladder, IF2 and quadrature nodes: high-impedance lumped
nodes, no impedance target, no driven edge, not marked RF because no line
impedance applies to them).

Expected statuses on this machine (ngspice-42, the packed KiCad 10.0.6
libraries, no kicad-cli; measured by ``tests/test_kr447_rx_backend.py``):
every ``spice.rf.*`` row of the five fixture networks PASS as "a network
verdict under confirmed model values (not a measured part)";
``rf.freq_plan`` NOT_VERIFIED (its margin rows PASS; the second-image
response row points to the fixture row ``image2`` and to the lab item
``ifb_image2_rejection``, so it never PASSes before a measurement, like every
receiver response row), ``rf.model_grounding`` / ``rf.regulatory_profile`` /
``rf.lab.*`` NOT_VERIFIED. ``si.rf_length`` is NOT_VERIFIED on this board:
``ai_eda.design.board.add_board`` builds ``ir.si`` without
``rf_length_fraction`` (a wave-1 gap a template cannot close; named in the
merge notes). The default board (the P9 companions; 100 parts, 92 x 64 mm)
adds the companions' five design-deck rows (all PASS), ``power.*`` and their
lab items (NOT_VERIFIED) and serves every requirement
(``review.requirements_vs_ir`` PASS); routed, it has no FAIL anywhere
(measured once on ngspice-42: 812 tracks, a 74 s confirm run).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable

from ai_eda.ir import Block as TopologyBlock
from ai_eda.ir import CircuitDomain, CircuitIR, MissingInformation, NetClass, NetKind, SimulationSetup, SpiceBinding, Topology, Traced
from ai_eda.ir.rf import RFBlock, RFDesign, RFPort, RFRegion
from ai_eda.tools.calc import radio
from ai_eda.tools.kicad.library import KicadLibrary

from ai_eda.design.base import (
    NO_RECORD,
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
from ai_eda.design.rf.blocks.if_backend import BLOCK_ID as IF_BLOCK_ID
from ai_eda.design.rf.blocks.if_backend import IfBackendBlock
from ai_eda.design.rf.blocks.power import PACK_CUTOFF_V, PACK_MAX_V
from ai_eda.design.rf.family import BUILDS, unserved_message
from ai_eda.design.rf.models import MODEL_VERDICT
from ai_eda.design.rf.parts import part_line
from ai_eda.design.rf.profile import profile_choices, profile_keys

if TYPE_CHECKING:
    from ai_eda.design.board import BoardContext, SIDeclarations
    from ai_eda.report.figures import Figure

#: the radio_build this template builds
RADIO_BUILD = "rx_backend"
#: the family's entry (needs / serves / board text)
BUILD = BUILDS[RADIO_BUILD]
#: why only 4 layers (kr447 design §2.0, decision 7A)
PLANE_REASON = "the RF lines need a reference plane (decision 7A: 4 layers)"
#: how each block of the board is named: references re-based by hundreds, internal nets prefixed (``power`` / ``rx_audio``: part P9's companions)
PREFIXES: dict[str, BlockPrefix] = {
    IF_BLOCK_ID: BlockPrefix(500, ""),
    "bench_io": BlockPrefix(0, ""),
    "power": BlockPrefix(100, "PWR_"),
    "rx_audio": BlockPrefix(400, "RXA_"),
}
#: the floorplan region of each block (x, y, w, h in mm from the board's top-left corner): choices ``floor.<block>.*``; the board is their bounding box
#: The IF back-end fills the left 50 mm; the RX audio block (volume, squelch, speaker) sits right of it and the RX power section below
#: that (92 x 64 mm with the default companions); ``bench_io`` is the stand-in's strip under the IF back-end (50 x 76 mm).
REGIONS: dict[str, tuple[float, float, float, float]] = {
    IF_BLOCK_ID: (0.0, 0.0, 50.0, 64.0),
    "rx_audio": (50.0, 0.0, 42.0, 42.0),
    "power": (50.0, 42.0, 42.0, 22.0),
    "bench_io": (0.0, 64.0, 50.0, 12.0),
}
#: the pack voltages the RX power section is designed for: above the confirmed cut-off (``power.pack_cutoff_v``), at most a full 2S pack
V_IN_RANGE: tuple[float, float] = (PACK_CUTOFF_V, PACK_MAX_V)
#: the audio bandwidths the RX audio block's low-pass is designed for (the stage-1 board's range: a voice channel)
BANDWIDTH_RANGE: tuple[float, float] = (2000.0, 4000.0)


# --------------------------------------------------------------------------- companions


class BenchInterfaceBlock(Block):
    """Two bench headers in place of the RX power and audio blocks: ``J1`` RX_5V / GND / MUTE in, ``J2`` DISC_OUT / RSSI / GND out.

    A stand-in, honest about being one: it supplies ``RX_5V`` from the bench
    (a rail port of ``bench.rx_5v``, so ``block.interface.RX_5V`` compares it
    with the IF back-end's), lets the bench drive ``MUTE`` and brings the
    audio and RSSI out. It reads no requirement: a board built with it serves
    ``input_voltage`` and ``audio_bandwidth`` only once the P9 blocks replace
    it (the template's plan note says so).
    """

    id = "bench_io"
    title = "bench interface: RX_5V / MUTE in, DISC_OUT / RSSI out (stand-in for the RX power and audio blocks)"
    interface_nets = ("RX_5V", "DISC_OUT", "RSSI", "MUTE")

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        v = b.choice("bench.rx_5v", 5.0, "V", "the bench supply on J1 for RX_5V (stand-in for the RX power section's LP38693DT-5.0 output)")
        j1 = b.part("header_3", "J1", "Conn_01x03", "bench header: RX_5V supply, GND, MUTE control in")
        j2 = b.part("header_3", "J2", "Conn_01x03", "bench header: DISC_OUT audio, RSSI, GND out")
        b.net("RX_5V", NetKind.POWER, j1.at("Pin_1"), "bench supply RX_5V")
        b.net("MUTE", NetKind.SIGNAL, j1.at("Pin_3"), "bench mute control")
        b.net("DISC_OUT", NetKind.ANALOG, j2.at("Pin_1"), "discriminator audio to the bench")
        b.net("RSSI", NetKind.ANALOG, j2.at("Pin_2"), "RSSI to the bench")
        b.net("GND", NetKind.GROUND, [*j1.at("Pin_2"), *j2.at("Pin_3")], "ground")
        for ref in ("J1", "J2"):
            b.bind(ref, SpiceBinding(exclude=True, exclude_reason="bench header: a connector, no electrical model", provenance=ctx.provenance(f"{ref} excluded")))
        b.result.ports = [RFPort(name="RX_5V", net="RX_5V", kind="rail", voltage_v=v, direction="out")]
        b.result.chain = ["J1", "J2"]
        b.result.notes.append("bench_io: two headers stand in for the RX power section and the RX audio block (part P9): input_voltage and audio_bandwidth "
                              "are read by those blocks, not by this stand-in")
        return b.done()


def default_companions() -> tuple[Block, ...]:
    """The RX power section and the RX audio block (part P9) the board composes: ``PowerBlock(modes=("rx",))`` and ``RxAudioBlock()``."""
    from ai_eda.design.rf.blocks.power import PowerBlock
    from ai_eda.design.rf.blocks.rx_audio import RxAudioBlock

    return (PowerBlock(modes=("rx",)), RxAudioBlock())


def out_of_range(inputs: dict[str, DesignInput]) -> str | None:
    """Why a stated companion input is outside the design range of the P9 blocks (``input_voltage``, ``audio_bandwidth``), else ``None``."""
    for key, (lo, hi), why in (
        ("input_voltage", V_IN_RANGE, f"the 2S Li-ion pack is used only above the confirmed cut-off {PACK_CUTOFF_V:.12g} V (the minimum input of the "
                                      f"LP38693DT-5.0), and a full 2S pack is {PACK_MAX_V:.12g} V (the P-FET gates see the whole pack "
                                      "[UNVERIFIED: AOS AO3401A datasheet])"),
        ("audio_bandwidth", BANDWIDTH_RANGE, "the RX low-pass's design range (a voice channel)"),
    ):
        inp = inputs.get(key)
        if inp is not None and not lo <= inp.traced.value <= hi:
            return f"{key} {inp.traced.value:.12g} {inp.traced.unit} ({inp.requirement.id}) is outside {lo:.12g}..{hi:.12g} {inp.traced.unit}: {why}"
    return None


# --------------------------------------------------------------------------- the template


def _refused(plan: Plan, why: str) -> Plan:
    plan.notes.append(f"template {plan.template} not proposed: {why}")
    return plan


class KR447RxBackendTemplate(Template):
    """``radio_build = rx_backend``: IF1 21.4 MHz U.FL input -> crystal ladder -> SA605D -> RSSI / mute / audio (see the module docstring)."""

    id = BUILD.template_id
    title = BUILD.title
    triggers = ("radio_build",)
    needs = ("modulation", "input_voltage")
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
                elif k == "modulation":
                    plan.questions.append(MissingInformation(key=k, rationale="template input",
                                                             question=f"The {self.title} template needs modulation: answer modulation=fm (the KR 447 MHz class is FM [UNVERIFIED])"))
                else:
                    plan.questions.append(MissingInformation(key=k, rationale="template input",
                                                             question=f"The {self.title} template needs {k} in V (the pack voltage the RX power section regulates): answer {k}=<value V> (e.g. {k}=\"7.4 V\")"))
            return _refused(plan, f"input(s) {missing} missing")
        modulation, _ = read_modulation(ir)
        if modulation != BUILD.modulation:
            return _refused(plan, f"modulation {modulation!r}: the KR 447 MHz licence-exempt class is FM telephony [UNVERIFIED: 「무선설비규칙」]; this board demodulates FM only")
        why = out_of_range(inputs)
        if why is not None:
            return _refused(plan, why)
        companions = self.companions()
        if not companions:
            return _refused(plan, ("the RX power section and the RX audio block (kr447 design §2.1, part P9) are not composed: this board is the IF back-end "
                                   "plus those two blocks, and a board without them would leave input_voltage and audio_bandwidth unserved "
                                   "(default_companions() composes them)"))
        plan.inputs = {k: inputs[k] for k in self.serves if k in inputs}
        profile = profile_choices(t, confirmed)
        shared = {c.key: tr for c, tr in profile}
        ctx = BlockContext(ir=ir, library=library, template_id=t, confirmed=confirmed, inputs=inputs, shared=shared)
        blocks: list[Block] = [IfBackendBlock(), *companions]
        results: list[BlockResult] = []
        try:
            for blk in blocks:
                prefix = PREFIXES.get(blk.id)
                if prefix is None:
                    raise TemplateRefusal(f"block {blk.id!r} has no reference / net prefix in PREFIXES")
                if blk.id not in REGIONS:
                    raise TemplateRefusal(f"block {blk.id!r} has no floorplan region in REGIONS")
                results.append(blk.build(ctx, prefix))
            board = merge_results(results, block_id=t)
        except TemplateRefusal as e:
            return _refused(plan, str(e))
        components, report = exclude_floating(board.components, board.nets, board.stimuli, t)
        bad = [f for f in report.unresolved if not f.ref.startswith("RV")]  # a pot's wiper keeps its DC path through the pot's own resistors
        if bad:
            return _refused(plan, "; ".join(f.reason for f in bad))
        region_choices: list[tuple[Choice, Traced]] = []
        rf_blocks: list[RFBlock] = []
        for r in results:
            vals = {}
            for axis, value in zip(("x", "y", "w", "h"), REGIONS[r.block_id]):
                key = f"floor.{r.block_id}.{axis}"
                desc = f"floorplan region of block {r.block_id}: {'left edge' if axis == 'x' else 'top edge' if axis == 'y' else 'width' if axis == 'w' else 'height'} (mm, board frame)"
                ch = Choice(key, desc, value, "mm")
                tr = Traced(value=value, unit="mm", provenance=choice_provenance(t, f"{key} = {value!r} mm: {desc}", confirmed))
                region_choices.append((ch, tr))
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
        plan.parts = [part_line(board.placed[c.ref]) for c in components if c.ref in board.placed]
        plan.nets = [f"{n.name} ({n.kind.value}): {', '.join(f'{p.component_ref}.{p.pin_number}' for p in n.pins)}" for n in board.nets]
        plan.simulation = self._simulation_lines(board, components, report.unresolved)
        plan.notes.extend(board.notes)
        topology = Topology(
            name=self.title, domains=[CircuitDomain.RF, CircuitDomain.ANALOG, CircuitDomain.POWER],
            rationale=("IF1 21.4 MHz (50 ohm U.FL) -> L-match -> C0-aware Butterworth crystal ladder -> L-match -> SA605D mixer with LO2 = IF1 - IF2, "
                       "450 kHz top-C IF2 filters, limiter, quadrature detector (tank to RX_5V), RSSI -> squelch -> MUTE, muted audio -> DISC_OUT. "
                       "The SA605D has no model: its passive networks are the RF fixtures, judged under confirmed model values"),
            provenance=ctx.provenance(f"selected by radio_build = {RADIO_BUILD}"),
            blocks=[TopologyBlock(id=r.block_id, function=r.title, domain=CircuitDomain.RF if r.block_id == IF_BLOCK_ID else CircuitDomain.ANALOG,
                                  component_refs=[c.ref for c in r.components], input_nets=[p.net for p in r.ports if p.direction == "in"],
                                  output_nets=[p.net for p in r.ports if p.direction == "out"], provenance=ctx.provenance(f"block {r.block_id}"))
                    for r in results],
        )
        changes = [DesignChange(description=f"topology: {self.title}", target="topology", operation="set", payload=topology, rationale=f"template {t} v{self.version}")]
        changes += [DesignChange(description=f"add {c.ref} ({c.description})", target="components", operation="append", payload=c) for c in components]
        changes += [DesignChange(description=f"add net {n.name}", target="nets", operation="append", payload=n) for n in board.nets]
        changes += [DesignChange(description=f"parameter {k}", target=f"parameters.{k}", operation="set", payload=v) for k, v in params.items()]
        if board.stimuli or board.analyses or board.expectations:
            changes.append(DesignChange(description="simulation setup", target="simulation", operation="set",
                                        payload=SimulationSetup(stimuli=board.stimuli, analyses=board.analyses, expectations=board.expectations)))
        changes += [DesignChange(description=f"constraint {c.id}", target="constraints", operation="append", payload=c) for c in board.constraints]
        changes.append(DesignChange(description="RF design: blocks, fixture networks, frequency plan, lab items", target="rf", operation="set", payload=rf,
                                    rationale=f"template {t}: {len(rf.networks)} fixture network(s), {len(rf.frequency_plan)} plan row(s), {len(rf.lab_items)} lab item(s)"))
        plan.changes = changes
        return plan

    @staticmethod
    def _simulation_lines(board: BlockResult, components, kept=()) -> list[str]:
        lines = []
        for nw in board.networks:
            rows = ", ".join(f"{e.id} ({e.quantity} at {float(e.at.value):.9g} Hz: "
                             + (f"{e.bound} {float(e.nominal.value):.6g}" if e.bound else f"{float(e.nominal.value):.6g} +/- {float(e.tol_abs.value):.6g}") + ")"  # type: ignore[union-attr]
                             for e in nw.expectations)
            lines.append(f"fixture {nw.id}: {len(nw.members)} member(s) between {', '.join(f'{p.name} ({p.kind})' for p in nw.ports)}; rows {rows}")
        lines.append(f"every fixture PASS is a {MODEL_VERDICT}, schematic level (no track, via or ground-return inductance)")
        if board.expectations:
            lines.append(f"design deck: {len(board.expectations)} expectation(s) of the companion blocks")
        else:
            lines.append("design deck: none (the SA605D has no model and the IF back-end's parts are fixture members only; the RX audio block brings the deck)")
        excluded = [c.ref for c in components if c.spice is not None and c.spice.exclude]
        lines.append(f"excluded from the design deck ({len(excluded)}): {', '.join(excluded)} (reasons in each part's SPICE binding)")
        lines += [f"reported, kept: {f.reason}" for f in kept]
        return lines

    # ------------------------------------------------------------------ board

    def si_declarations(self, ir: CircuitIR, ctx: BoardContext) -> SIDeclarations:
        """``IF50`` (the IF1 input at the system impedance, an RF class at IF1) and ``IF_HIZ`` (the ladder / IF2 / quadrature nodes: no target, no edge)."""
        from ai_eda.design.board import SIDeclarations

        out = SIDeclarations()
        z0 = ctx.params.get("ifb.z0") or ctx.params.get("rf.z0")
        if1 = ctx.params.get("rf.if1")
        if z0 is None or if1 is None:
            return out
        tol = ctx.choice("si.if50_z0_tol", 0.10, None, "tolerance of the IF50 class's impedance (10 %)")
        out.classes.append(NetClass(
            name="IF50", nets=["IF1"], target_z0_ohm=z0, z0_tol_rel=tol, rf_frequency_hz=if1,
            description="the 50 ohm IF1 input (U.FL to the input L-match): a line at the system impedance over the In1.Cu ground plane",
            provenance=ctx.structural("IF1 input line at the system impedance"),
        ))
        hiz = if_backend_hiz_nets(ctx.params)
        out.classes.append(NetClass(
            name="IF_HIZ", nets=hiz,
            description=("the crystal ladder, the SA605 mixer input, IF2 filter, limiter, quadrature and LO2 nodes: high-impedance lumped nodes at 21.4 MHz / "
                         "450 kHz, short against lambda_g / 10 (about 778 mm at 21.4 MHz) - no impedance target, no driven edge, so no critical length and no line check"),
            provenance=ctx.structural("IF back-end high-impedance nodes"),
        ))
        out.lines.append("IF50 / IF_HIZ: the ladder's 845 ohm side is not a 50 ohm line (critic: LAD_IN moved out of IF50); si.rf_length needs "
                         "ir.si.rf_length_fraction, which the board module does not write yet")
        return out

    # ------------------------------------------------------------------ report views

    def _n(self, ir: CircuitIR, key: str) -> float | None:
        return parameter_value(ir, key)

    def theory(self, ir: CircuitIR) -> list[TheorySection]:
        p = lambda k: self._n(ir, k)  # noqa: E731
        order = p("ifb.lad.n")
        n_int = int(order) if order is not None else None
        fs, f0, bw_d, bw, rend = p("ifb.lad.xtal_fs"), p("ifb.lad.f0"), p("ifb.lad.bw_design"), p("rf.if_bw"), p("ifb.lad.r_end")
        cm, c0, rm, lm = p("model.xtal21.cm"), p("model.xtal21.c0"), p("model.xtal21.rm"), p("ifb.lad.xtal_lm")
        if1, if2, lo2, img2 = p("rf.if1"), p("rf.if2"), p("ifb.lo2.f"), p("ifb.image2")
        spacing = p("ifb.channel_spacing") or p("kr447.channel_raster")
        s21 = {k: p(f"ifb.lad.s21.{k}") for k in ("f0", "pass_lo", "pass_hi", "acs_lo", "acs_hi")}
        c_at_if1 = p("ifb.lad.center_at_if1")
        kappa = None
        bound6 = None
        if cm is not None and c0 is not None and if1 is not None:
            g6 = [2.0 * math.sin((2 * k - 1) * math.pi / 12.0) for k in range(1, 7)]
            k6 = [1.0 / math.sqrt(g6[i] * g6[i + 1]) for i in range(5)]
            kappa = max((k6[i - 1] if i > 0 else 0.0) + (k6[i] if i < 5 else 0.0) for i in range(6))
            bound6 = if1 * 6e-15 / (4.0 * c0 * kappa)
        couple = [p(f"ifb.lad.c_couple.{i}") for i in range(1, (n_int or 1))]
        mesh = {i: p(f"ifb.lad.c_mesh.{i}") for i in range(1, (n_int or 0) + 1) if p(f"ifb.lad.c_mesh.{i}") is not None}
        q_if1, q_if2 = p("model.l_q.if1"), p("model.l_q.if2")
        z0 = p("ifb.z0") or p("rf.z0")
        lin, cin, lout, cout, rin = p("ifb.match_in.l"), p("ifb.match_in.c"), p("ifb.match_out.l"), p("ifb.match_out.c"), p("model.sa605.rf_in_r")
        q_in = math.sqrt(rend / z0 - 1.0) if rend and z0 and rend > z0 else None
        q_out = math.sqrt(max(rin, rend) / min(rin, rend) - 1.0) if rin and rend and rin != rend else None
        match_loss = (4.343 * (q_in + q_out) / q_if1) if q_in is not None and q_out is not None and q_if1 else None
        l2, bw2, tap, cc2, sh2 = p("ifb.if2.l"), p("ifb.if2.bw"), p("ifb.if2.c_tap"), p("ifb.if2.c_couple"), p("ifb.if2.c_shunt.1")
        s21_2, rel2_lo, rel2_hi = p("ifb.if2.s21"), p("ifb.if2.rel_lo"), p("ifb.if2.rel_hi")
        lq, cfix, cq, ctq, rp, qq = p("ifb.quad.l"), p("ifb.quad.c_fixed"), p("ifb.quad.c_q"), p("ifb.quad.c_trim"), p("ifb.quad.r_p"), p("ifb.quad.q")
        ph = {k: p(f"ifb.quad.phase.{k}") for k in ("lo", "mid", "hi")}
        dev = p("ifb.deviation") or p("kr447.max_deviation")
        c_osc, c_load = p("ifb.lo2.c_osc"), p("ifb.lo2.c_load")
        tau = p("ifb.rssi.tau")
        mesh_text = ", ".join(f"메시 {i}: {quantity(v, 'F')}" for i, v in mesh.items()) or "기록 없음"
        couple_text = ", ".join(quantity(v, "F") for v in couple) or "기록 없음"
        return [
            TheorySection("개요: IF 백엔드 시험 보드", (
                "신호 흐름: IF1 입력(U.FL, 50 Ω) → L 정합 → 수정 래더 필터 → L 정합 → SA605D 혼합기(LO2 수정 발진) → 450 kHz IF2 필터 2개 → "
                "IF 증폭기 → 리미터 → 직교(quadrature) 검파기 → 음성(DISC_OUT), RSSI → 스컬치 → MUTE.\n\n"
                "전원(2S 팩 → 퓨즈 → 주 P-FET 스위치 → LP38693DT-5.0 RX_5V → LP5907 RX_3V3, 참조 1xx)과 RX 오디오(디엠퍼시스 + 300–3000 Hz 대역통과 → 볼륨 → "
                "LM386 스피커 앰프, RSSI 스켈치 비교기 → MUTE, 참조 4xx)는 1단계 오디오/PTT 보드와 같은 블록을 그대로 씁니다(송신부 없음: RX 레일은 항상 켜짐). "
                "설계 덱은 이 두 블록의 것입니다: 주 스위치의 동작점과 DISC_OUT → 볼륨 입력의 교류 응답, 스켈치 문턱 분압.\n\n"
                "SA605D 는 SPICE 모델이 없어 모든 넷리스트에서 제외됩니다. 대신 SA605D 가 보는 수동 회로망(래더, IF2 필터, 직교 탱크)을 각각 RF 픽스처로 잘라내어 "
                "ngspice 교류 해석으로 판정합니다. 그 PASS 는 '확인된 모델값 아래에서의 회로망 판정(측정된 부품이 아님)'이며, 트랙·비아·접지 귀환 인덕턴스가 없는 "
                "회로도 수준의 판정입니다. 수정 모델값·SA605D 포트 저항·인덕터 Q 는 모두 검증되지 않은(UNVERIFIED) 선택값이고, KR 447 MHz 규제 수치(kr447.*)도 "
                "법령 원문으로 근거를 두지 않은 자리표시값입니다. 송신 전 KC 적합성평가가 필요합니다(이 보드는 수신 전용)."
            )),
            TheorySection("주파수 계획", (
                f"- IF1 = {quantity(if1, 'Hz')}, IF2 = {quantity(if2, 'Hz')} (선택값).\n"
                f"- LO2 = IF1 − IF2 = {quantity(lo2, 'Hz')} (저측 LO, `calc.rf.superhet.lo`).\n"
                f"- SA605 혼합기의 2차 영상 = 2·LO2 − IF1 = {quantity(img2, 'Hz')} (`calc.rf.superhet.image`): 혼합기 앞의 래더가 제거합니다(픽스처 행 image2).\n"
                f"- 리미터 출력의 IF2 고조파 중 IF1 에 가장 가까운 것은 47·IF2 = {quantity(p('ifb.if2.harm_below'), 'Hz')}, 48·IF2 = {quantity(p('ifb.if2.harm_above'), 'Hz')} "
                f"(`calc.rf.harmonic`): 채널 간격 {quantity(spacing, 'Hz')} 보다 멀리 떨어져 래더가 거부합니다(rf.freq_plan 여유 행).\n"
                "- 455 kHz 대신 450 kHz 를 쓰는 이유: 47 × 455 kHz = 21.385 MHz 가 IF1 에서 15 kHz 밖에 떨어지지 않습니다."
            )),
            TheorySection("수정 래더 필터: C0 를 고려한 Dishal 설계", (
                f"하측파대(lower-sideband) 래더: 수정 {number(order, 3)}개를 직렬로, 결합 커패시터를 병렬(접지)로, 각 메시에 직렬 조정 커패시터를 둡니다(결합 합이 가장 큰 메시는 제외). "
                "고전적인 EMRFD 식 C_ij = C_m f_0 / (k_ij BW) 는 홀더 용량 C0 를 무시하여, 이 설계의 초기안(C_m 6 fF, C0 4 pF)으로 7.5 kHz 를 목표로 하면 "
                "ngspice-42 에서 3.4 kHz 가 나왔습니다. 여기서는 같은 Dishal 방법을 리액턴스 기울기 파라미터로 적용합니다(Matthaei·Young·Jones 8.02, EMRFD 3장):\n\n"
                "    X_c = X_m / (1 − ω C0 X_m),  X_m = ω L_m − 1/(ω C_m)\n"
                "    x = (ω0/2) dX_c/dω + X_c/2       (메시의 리액턴스 기울기)\n"
                "    K_ij = w x k_ij,  C_ij = 1/(ω0 K_ij),  R_end = w x / g_1   (w = BW/f0, k_ij = 1/sqrt(g_i g_j))\n\n"
                f"**C_m 을 {_ff(cm)} 로 둔 이유.** C0 때문에 이 래더의 대역폭에는 상한 BW_max ≈ f_s C_m / (4 C0 κ) 가 있습니다. 6 fF 이면 수정 6개일 때 "
                f"{quantity(bound6, 'Hz')} (κ = {number(kappa, 4)}) 로 7.5 kHz 래더가 존재하지 않아 계산기가 설계를 거부합니다. 그래서 AT-컷 기본파 수정의 전형적인 용량비 C0/C_m ≈ 250 "
                f"(C0 {quantity(c0, 'F')} → C_m {_ff(cm)}) 을 모델값으로 택했습니다(검증되지 않음: 주문한 수정을 G3UUR 법으로 측정해 바꿔야 하는 조달 사양). "
                "계산기나 판정 기준은 바꾸지 않았습니다.\n\n"
                f"**수정 {number(order, 3)}개인 이유.** C_m 16 fF 로는 6개짜리 버터워스 래더가 통과대역 행(±IF_BW/2 에서 −3.5 dB 이상)과 인접채널 행(±채널 간격에서 −40 dB 이하)을 "
                "동시에 만족하지 못합니다(하측파대 래더는 낮은 쪽 스커트가 완만하여 최선이 −38.1 dB). 7개면 둘 다 여유 있게 만족합니다.\n\n"
                f"**사전 왜곡(pre-distortion).** 수정의 운동 저항 R_m = {quantity(rm, 'ohm')} 이 통과대역을 좁히므로 설계 대역폭을 {quantity(bw_d, 'Hz')} 로 넓게 잡고, "
                f"판정은 확정된 IF 대역폭 {quantity(bw, 'Hz')} 에서 정확한 응답(`calc.crystal.ladder.s21_db`, R_m 포함)으로 합니다.\n\n"
                f"**중심 주파수.** 하측파대 래더의 통과대역은 직렬공진보다 위에 있습니다. 수정을 IF1 에 두면 중심이 {quantity(c_at_if1, 'Hz')} 가 되어 IF1 신호가 아래쪽 스커트에 걸립니다. "
                f"그래서 수정의 직렬공진을 f_s = 2·IF1 − f_c(IF1) = {quantity(fs, 'Hz')} 로 지정합니다(중심을 IF1 에 대해 되접은 값; `calc.rf.superhet.image` 의 2·LO − f 를 이 반사에 사용). "
                f"그 결과 실현 중심 f0 = {quantity(f0, 'Hz')} (`calc.crystal.ladder.center`) 이며, 픽스처 행 if1_centre 가 IF1 이 평탄한 통과대역 안에 있음을 판정합니다.\n\n"
                f"| 항목 | 값 |\n|---|---|\n| 운동 인덕턴스 L_m (`calc.rf.lc.l_for_resonance`) | {quantity(lm, 'H')} |\n| 종단 저항 R_end | {quantity(rend, 'ohm')} |\n"
                f"| 결합 커패시터 | {couple_text} |\n| 메시 조정 커패시터 | {mesh_text} |\n"
                f"| S21(f0) (R_m 포함, R_end 종단) | {number(s21['f0'], 4)} dB |\n| 통과대역 끝 f0 ∓ {quantity(_half(bw), 'Hz')} (f0 기준) | {number(_rel(s21['pass_lo'], s21['f0']), 4)} / {number(_rel(s21['pass_hi'], s21['f0']), 4)} dB |\n"
                f"| 인접채널 f0 ∓ {quantity(spacing, 'Hz')} (f0 기준) | {number(_rel(s21['acs_lo'], s21['f0']), 4)} / {number(_rel(s21['acs_hi'], s21['f0']), 4)} dB |\n\n"
                f"**손실.** R_m = {quantity(rm, 'ohm')} 이면 래더 자체가 중심에서 약 {number(_neg(s21['f0']), 3)} dB 를 잃으므로 설계 초기안의 's21 at f_0 ≥ −4 dB' 는 성립할 수 없습니다. "
                "그래서 이 행은 정확한 래더 응답을 공칭값으로 하고 허용오차 1 dB 로 판정합니다(손실 있는 회로망의 손실 행에 대한 설계 문서의 규칙, §2.3)."
            )),
            TheorySection("L 정합", (
                f"입력: {quantity(z0, 'ohm')} → R_end {quantity(rend, 'ohm')}, 저역통과형(직렬 L 은 저저항 쪽, 병렬 C 는 고저항 쪽). Q = sqrt(R_hi/R_lo − 1) = {number(q_in, 4)}, "
                f"L = Q·R_lo/ω = {quantity(lin, 'H')}, C = Q/(ω R_hi) = {quantity(cin, 'F')} (`calc.rf.lmatch.lowpass.*`).\n"
                f"출력: R_end → SA605 RF_IN 모델 저항 {quantity(rin, 'ohm')}: Q = {number(q_out, 4)}, L = {quantity(lout, 'H')}, C = {quantity(cout, 'F')}.\n\n"
                f"인덕터 Q {number(q_if1, 3)} (모델값) 에서 두 정합의 손실은 대략 4.343·(Q_in + Q_out)/Q_u ≈ {number(match_loss, 3)} dB 로, if1_filter 의 손실 행 허용오차(1 dB) 안에 있습니다. "
                "래더만의 응답은 R_end 종단의 별도 픽스처 if1_ladder 가 계산기와 0.2 dB 이내로 비교합니다."
            )),
            TheorySection("450 kHz IF2 필터(상단 결합 2단)", (
                f"공진기 L = {quantity(l2, 'H')}, 대역폭 {quantity(bw2, 'Hz')}, 양단 1.5 kΩ 모델 포트(`model.sa605.port_r`). Butterworth g 로 Q_e = g_1 f0/BW, "
                "R_p = Q_e ω0 L, 용량성 탭 C_s = 1/(ω0 sqrt(R_p R_t − R_t²)), 결합 C_k = k (BW/f0) C_res, 병렬 C_i = C_res − 결합 − 탭 등가 (Dishal 1949; Zverev 1967).\n\n"
                f"| 항목 | 값 |\n|---|---|\n| 탭 커패시터 | {quantity(tap, 'F')} |\n| 결합 커패시터 | {quantity(cc2, 'F')} |\n| 공진 커패시터 | {quantity(sh2, 'F')} |\n"
                f"| S21(IF2) (Q_u {number(q_if2, 3)}) | {number(s21_2, 4)} dB |\n| IF2 ∓ 100 kHz (IF2 기준) | {number(rel2_lo, 4)} / {number(rel2_hi, 4)} dB |\n\n"
                "상단 결합 회로망은 비대칭이라(위쪽 스커트가 약함) 대칭 협대역 식이 아니라 정확한 회로망 응답(`calc.rf.resonator.top_c.*`)을 공칭값으로 씁니다. "
                "채널 선택도는 래더가 담당하고, 이 필터는 혼합기 산물(IF1, LO2, IF1 + LO2)을 정리합니다."
            )),
            TheorySection("직교(quadrature) 검파기", (
                f"리미터 출력이 결합 커패시터 C_q = {quantity(cq, 'F')} 를 거쳐 탱크 L // (C_fixed + C_trim) // R_p 로 들어갑니다(탱크는 RX_5V 로 귀환: 교류 접지). "
                f"L = {quantity(lq, 'H')} (1 nF 와 IF2 에서 공진, `calc.rf.lc.l_for_resonance`), C_fixed = C_res − C_q − C_trim = {quantity(cfix, 'F')}, "
                f"C_trim(중간 위치) = {quantity(ctq, 'F')}, R_p = {quantity(rp, 'ohm')} 로 Q = R_p/(ωL) = {number(qq, 4)} (`calc.rf.q_parallel`).\n\n"
                "검파기 출력은 리미터 출력과 위상 이동된 신호의 곱이므로 위상이 90° 일 때 0 입니다. C_q 가 탱크에 더해져 공진이 내려가므로 C_q 만큼 뺀 용량으로 공진시켜 "
                f"IF2 에서 90° 에 가깝게 둡니다. 정확한 위상(`calc.rf.quad.phase`, 소스 저항과 C_q 포함): IF2 − Δf / IF2 / IF2 + Δf = "
                f"{number(ph['lo'], 5)} / {number(ph['mid'], 5)} / {number(ph['hi'], 5)} ° (Δf = {quantity(dev, 'Hz')}). 90° 와의 차이는 트리머로 정렬합니다."
            )),
            TheorySection("LO2 발진기, RSSI, 뮤트", (
                f"LO2 수정({quantity(lo2, 'Hz')})은 OSC_IN 에서 접지로, 콜피츠 분압 커패시터 두 개(OSC_IN–OSC_OUT, OSC_OUT–GND)가 각각 {quantity(c_osc, 'F')} "
                f"(`calc.crystal.c_for_load`: C1 = C2 = 2(C_L − C_stray), C_L = {quantity(c_load, 'F')}). 발진기 자체는 시뮬레이션하지 않습니다(SA605D 모델 없음).\n\n"
                f"RSSI_OUT 은 전류 출력이며 부하 저항과 커패시터로 전압이 됩니다(시정수 τ = {quantity(tau, 's')}, `calc.rc.tau`). RSSI 전압은 스컬치 비교기로, 비교기 출력은 MUTE_INPUT 으로 갑니다. "
                "RSSI 기울기와 스컬치 임계값은 실험실 항목입니다."
            )),
        ]

    def theory_figures(self, ir: CircuitIR) -> list[Figure]:
        """The ladder's exact |S21| around f0 (R_m included, R_end terminations) with the passband and adjacent-channel guides; the quadrature phase around IF2."""
        from ai_eda.design.templates import THEORY_CURVE_NOTE, _curve_figure

        p = lambda k: self._n(ir, k)  # noqa: E731
        out: list[Figure] = []
        n, bw, fs, cm, c0, rm, f0 = (p(k) for k in ("ifb.lad.n", "ifb.lad.bw_design", "ifb.lad.xtal_fs", "model.xtal21.cm", "model.xtal21.c0", "model.xtal21.rm", "ifb.lad.f0"))
        if_bw, spacing = p("rf.if_bw"), p("ifb.channel_spacing") or p("kr447.channel_raster")
        if None not in (n, bw, fs, cm, c0, rm, f0, if_bw, spacing):
            try:
                xs = [f0 - 20e3 + 40e3 * i / 400 for i in range(401)]
                ys = [radio.crystal_ladder_s21_db_value(n, bw, fs, cm, c0, rm, x) for x in xs]
            except ValueError:
                xs, ys = [], []
            if xs:
                caption = (f"'수정 래더 필터' 절의 설계 래더(수정 {number(n, 3)}개, 설계 대역폭 {quantity(bw, 'Hz')}, f_s = {quantity(fs, 'Hz')}, C_m {quantity(cm, 'F')}, C0 {quantity(c0, 'F')}, "
                           f"R_m {quantity(rm, 'ohm')})의 정확한 S21 (`calc.crystal.ladder.s21_db`, R_end 종단). 음영 = 판정 통과대역 f0 ± IF_BW/2, 안내선 = 인접채널 f0 ± {quantity(spacing, 'Hz')}. "
                           f"{THEORY_CURVE_NOTE}")
                out.append(_curve_figure(
                    "theory_ladder_s21", "수정 래더 S21 (R_m 포함)", caption, [("S21", xs, ys)], x_label="주파수 (Hz)", y_label="S21 (dB)",
                    bands=[("x", f0 - if_bw / 2, f0 + if_bw / 2, f"통과대역 {quantity(if_bw, 'Hz')}"), ("x", f0 - spacing, f0 - spacing, "인접채널 −"),
                           ("x", f0 + spacing, f0 + spacing, "인접채널 +")],
                ))
        if2, lq, cq, cfix, ctq, rp, q_u, r_s = (p(k) for k in ("rf.if2", "ifb.quad.l", "ifb.quad.c_q", "ifb.quad.c_fixed", "ifb.quad.c_trim", "ifb.quad.r_p", "model.l_q.if2", "model.sa605.lim_out_r"))
        if None not in (if2, lq, cq, cfix, ctq, rp, q_u, r_s):
            try:
                xs = [if2 - 10e3 + 20e3 * i / 200 for i in range(201)]
                ys = [radio.quad_phase_deg(x, r_s, cq, lq, q_u, if2, cfix, ctq, rp) for x in xs]
            except ValueError:
                xs, ys = [], []
            if xs:
                caption = (f"'직교 검파기' 절의 위상 arg V(QUAD)/V(LIMITER_OUT) (`calc.rf.quad.phase`, 소스 {quantity(r_s, 'ohm')}, C_q {quantity(cq, 'F')}, L {quantity(lq, 'H')}, "
                           f"R_p {quantity(rp, 'ohm')}). 안내선 = 90° (검파기 출력 0). {THEORY_CURVE_NOTE}")
                out.append(_curve_figure("theory_quad_phase", "직교 탱크 위상", caption, [("위상", xs, ys)], x_label="주파수 (Hz)", y_label="위상 (°)",
                                         bands=[("y", 90.0, 90.0, "90°")]))
        return out

    def part_notes(self, ir: CircuitIR) -> dict[str, PartNote]:
        from ai_eda.design.rf.t_audio_ptt import stage1_part_notes, stage1_refs

        p = lambda k: self._n(ir, k)  # noqa: E731
        stage1 = stage1_refs(ir)  # the P9 companions' parts: the stage-1 board's notes (their descriptions would match the IF back-end's patterns)
        out: dict[str, PartNote] = stage1_part_notes(ir, stage1)
        fs, cm, c0, rm = p("ifb.lad.xtal_fs"), p("model.xtal21.cm"), p("model.xtal21.c0"), p("model.xtal21.rm")
        for c in ir.components:
            if c.ref in stage1:
                continue
            d = c.description
            if "IF1 21.4 MHz input" in d:
                out[c.ref] = PartNote("IF1 21.4 MHz 입력 커넥터(신호 발생기 또는 프런트엔드의 IF1 출력)", "U.FL: 벤치 케이블 연결용; 50 Ω 선로(IF50 클래스)로 L 정합에 연결",
                                      ["50 Ω 동축 커넥터", "IF1 에서 삽입손실 무시 가능"], [unverified("Hirose U.FL-R-SMT-1(10)")])
            elif "ladder crystal" in d:
                out[c.ref] = PartNote(
                    "수정 래더 필터의 공진기(정합된 세트)", f"C0 를 고려한 Dishal 래더: 직렬공진 {quantity(fs, 'Hz')} 로 지정해 통과대역 중심을 IF1 에 둠",
                    [f"직렬공진 {quantity(fs, 'Hz')} (세트 내 편차 ±50 Hz 이하 권장)", f"C_m ≥ {quantity(cm, 'F')}, C0 ≤ {quantity(c0, 'F')}", f"R_m ≤ {quantity(rm, 'ohm')}",
                     "모든 수치는 검증되지 않은 모델값: 주문 후 G3UUR 법으로 측정"],
                    [unverified("21.4 MHz 기본파 HC-49/SD 수정 맞춤 주문", "필터용 정합 세트")])
            elif "ladder coupling capacitor" in d or "ladder mesh" in d:
                trim = c.symbol is not None and c.symbol.name == "C_Trim"
                out[c.ref] = PartNote(
                    "래더의 " + ("양 끝 메시 조정 트리머" if trim else "결합/메시 조정 커패시터"), "`calc.crystal.ladder.*` 가 계산한 값 그대로(E 계열 반올림 없음)",
                    ["NP0/C0G 유전체", "±1 % 이하 또는 선별", f"값 {c.value}F"] + (["트리머 범위가 계산값을 포함할 것"] if trim else []),
                    [unverified("Murata TZB4 트리머" if trim else "0603 NP0 1 %")])
            elif "L-match" in d:
                out[c.ref] = PartNote("래더 입출력 L 정합 소자", "`calc.rf.lmatch.lowpass.*`: 저역통과형 L 정합",
                                      [f"값 {c.value}", "인덕터: 21.4 MHz 에서 Q ≥ 30 (모델값)", "커패시터: NP0"], [unverified("0603 권선형 인덕터 / NP0 커패시터")])
            elif "FM IF system" in d:
                out[c.ref] = PartNote("FM IF 시스템(혼합기, LO2 발진기, IF 증폭기, 리미터, 직교 검파기, RSSI, 뮤트)",
                                      "라이브러리 핀 이름으로 배선; SPICE 모델이 없어 모든 넷리스트에서 제외, 포트 저항은 모델값",
                                      ["VCC 범위가 RX_5V 포함 (약 4.5–8 V, 검증되지 않음)", "혼합기 입력 25 MHz 이상 동작 (검증되지 않음)"],
                                      [unverified("NXP SA605D"), unverified("NXP SA615D", "핀 호환 여부 확인 필요")])
            elif "LO2 crystal" in d:
                out[c.ref] = PartNote("LO2 수정(20.95 MHz, SA605 콜피츠 발진기)", f"LO2 = IF1 − IF2 = {quantity(p('ifb.lo2.f'), 'Hz')}",
                                      [f"부하 용량 {quantity(p('ifb.lo2.c_load'), 'F')} 에서의 주파수로 지정", "허용오차: frequency_tolerance 요구사항(있으면)"],
                                      [unverified("20.950 MHz HC-49/SD 맞춤 주문")])
            elif "Colpitts" in d:
                out[c.ref] = PartNote("LO2 콜피츠 분압 커패시터", "`calc.crystal.c_for_load`: C1 = C2 = 2(C_L − C_stray)", ["NP0", f"값 {c.value}F"], [unverified("0603 NP0 5 %")])
            elif "IF2 filter" in d:
                out[c.ref] = PartNote("450 kHz IF2 상단 결합 필터 소자", "`calc.rf.resonator.top_c.*` 계산값", [f"값 {c.value}", "인덕터 Q ≥ 50 @ 450 kHz (모델값)", "커패시터 NP0 1 %"],
                                      [unverified("1210 권선형 인덕터 / 0603 NP0")])
            elif "quadrature" in d:
                out[c.ref] = PartNote("직교 검파기 회로망 소자", "90° 위상을 IF2 에 두는 탱크(`calc.rf.quad.phase`)", [f"값 {c.value}", "인덕터 Q ≥ 50 @ 450 kHz (모델값)"],
                                      [unverified("1210 인덕터 / Murata TZB4 트리머 / 0603 저항")])
            elif "decoupling" in d:
                out[c.ref] = PartNote("SA605 디커플링", "SA605 응용 회로의 관례값 100 nF (검증되지 않음)", ["X7R 이상", "전압 정격 ≥ 10 V"], [unverified("0603 X7R 100 nF")])
            elif "RSSI" in d:
                out[c.ref] = PartNote("RSSI 전류-전압 변환 / 평활", f"시정수 {quantity(p('ifb.rssi.tau'), 's')} (`calc.rc.tau`)", [f"값 {c.value}"], [unverified("0603 1 %")])
            elif "test point" in d:
                out[c.ref] = PartNote("정렬용 테스트 포인트", d, ["패드 1 mm"], [])
            elif "bench header" in d:
                out[c.ref] = PartNote("벤치 인터페이스 헤더(RX 전원·오디오 블록 대체)", d, ["2.54 mm 핀 헤더"], [unverified("일반 1x3 핀 헤더")])
        return out


def if_backend_hiz_nets(params: dict[str, Traced], net_prefix: str = "") -> list[str]:
    """The IF back-end's high-impedance node nets (the ``IF_HIZ`` class) as :class:`IfBackendBlock` names them, from its parameters (ladder order, tuned meshes)."""
    order_t = params.get("ifb.lad.n")
    if order_t is None:
        return []
    order = int(order_t.value)
    names = ["LAD_IN", *[f"LAD_N{i}" for i in range(1, order)], "LAD_OUT"]
    names += [f"LAD_M{i}" for i in range(1, order + 1) if f"ifb.lad.c_mesh.{i}" in params]
    names += ["MIX_IN", "MIX_OUT", "F2A_R1", "F2A_R2", "IFA_IN", "IFA_OUT", "F2B_R1", "F2B_R2", "LIM_IN", "LIM_OUT", "QUAD", "OSC_B", "OSC_E"]
    return [f"{net_prefix}{n}" for n in names]


def _ff(x: float | None) -> str:
    """A crystal's motional capacitance in femtofarads (the report's SI prefixes stop at pico)."""
    return NO_RECORD if x is None else f"{x * 1e15:.4g} fF"


def _half(x: float | None) -> float | None:
    return None if x is None else x / 2.0


def _rel(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def _neg(x: float | None) -> float | None:
    return None if x is None else -x


__all__ = [
    "BANDWIDTH_RANGE",
    "BUILD",
    "PLANE_REASON",
    "PREFIXES",
    "RADIO_BUILD",
    "REGIONS",
    "V_IN_RANGE",
    "BenchInterfaceBlock",
    "KR447RxBackendTemplate",
    "default_companions",
    "if_backend_hiz_nets",
    "out_of_range",
]
