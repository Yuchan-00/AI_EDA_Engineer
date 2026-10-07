"""The ``kr447_audio_ptt`` template: stage 1 of the KR 447 MHz walkie-talkie family - power, PTT sequencer / TX interlock, TX and RX audio, no RF.

Selected by the categorical requirement ``radio_build = audio_ptt`` (never by
a number), needing ``input_voltage`` (the 2S pack) and ``modulation`` (FM: the
licence-exempt 447 MHz class is FM telephony [UNVERIFIED]), serving
``frequency_deviation`` and ``audio_bandwidth`` (the TX limiter / integrator
and the RX filters are designed from them; the profile's maximum deviation
and 3 kHz are the choices when they are not stated) and ``tx_timeout`` (the
4060 time-out is built only when it is stated). Every other confirmed design
requirement refuses the board with the family's closed-world sentence naming
the build that serves it (``carrier_frequency is served by radio_build=...``).

The board (kr447 design §2.1 with the decisions 3A / 4B and the critic2
singles) is four blocks of :mod:`ai_eda.design.rf.blocks` re-based by hundreds
(power 1xx, ptt 2xx, tx_audio 3xx, rx_audio 4xx) plus the bench headers
J402-J404 (``DISC_OUT`` / ``RSSI``, ``PM_DRIVE`` / ``PA_ON``, ``MUTE`` /
``PA_PD``), composed by :func:`~ai_eda.design.rf.blocks.base.merge_results`
and cleaned of dead SPICE branches by
:func:`~ai_eda.design.rf.blocks.base.exclude_floating` (the volume pot's wiper,
which reaches only the unmodelled LM386, is the one reported multi-terminal
node: the pot's own resistors give it a DC path). Every KR regulatory number
is an UNVERIFIED ``kr447.*`` placeholder (the family's profile), every
modelling number an UNVERIFIED ``model.*`` value - all rows of the
``confirm_design`` table. The layer policy builds 2 or 4 layers and defaults
to 4 (one stack across the stages); the floorplan regions are choices
(``floor.*``) the RF floorplan placer packs, 74 x 82 mm (the design's
50 x 60 mm does not hold the ~145 parts' footprints with the shelf packer's
1 mm spacing and 2 mm edge margin; the power region grew 4 mm for the
LM1117's electrolytic output capacitor C105, and the TX audio region 1 mm -
taken from the PTT region below it - for the through-hole microphone MK301,
``Sensor_Audio:POM-2244P-C3310-2-R``, whose 6.5 mm courtyard is 2 mm wider
than the SMT capsule it replaced; the symbol's pin 1 "-" / pin 2 "+" land on
pads 1 / 2, and which terminal is the case is UNVERIFIED against the PUI
Audio datasheet).

Edge placement is a deviation from kr447 design §2.1 ("connectors on the
bottom edge; PTT and pots on the top edge"): the placer packs each block's
parts from its region's top-left corner in chain order, so a user-facing
part lands wherever its block's region and the packing put it. Measured on
both builds (74 x 82 mm, the KiCad 10.0.6 footprints, 2 mm edge margin):
``RV401`` (volume) and ``MK301`` sit on the top edge by packing order;
``RV402`` (squelch) is 16.3 mm below the top edge; ``SW201`` (PTT) is in the
board's interior, 37.5 mm below the top edge; ``J101`` (pack) is on the
left edge, 34 mm above the bottom edge; ``J401`` (speaker) is in the
interior, 60.2 mm above the bottom edge; only the bench headers J402-J404
are on the bottom edge. Moving them is left to manual placement in KiCad
(a separate edge region would take the parts out of their blocks);
``tests/test_kr447_audio_ptt.py`` pins these positions, so a layout change
must restate them.

What the design deck proves is principle under the confirmed model values:
the rail sequencing, the settle delay and the PA supply switch, the
low-pack inhibit's input network, the audio chain's responses, the
limiter / splatter filter at +20 dB overdrive and the integrator. No IC is
simulated; ``rf.deviation`` stays NOT_VERIFIED (K_pm is ``model.k_pm``);
``rf.model_grounding`` / ``rf.regulatory_profile`` / ``rf.lab.*`` /
``power.*`` never PASS here. Transmitting needs KC conformity assessment
first; this board does not transmit.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ai_eda.ir import (
    Block,
    CircuitDomain,
    CircuitIR,
    Constraint,
    ConstraintKind,
    MissingInformation,
    NetClass,
    SimulationSetup,
    Traced,
    Topology,
)
from ai_eda.ir.rf import RFBlock, RFDesign, RFRegion
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
    unusable_remedy,
    with_remedy,
)
from ai_eda.design.inputs import MODULATION_ALIASES, DesignInput, canonical_key, present_keys, read_modulation, read_radio_build
from ai_eda.design.library_parts import TemplateRefusal
from ai_eda.design.rf.blocks.base import Block as RFBlockBuilderBase
from ai_eda.design.rf.blocks.base import BlockBuilder, BlockContext, BlockPrefix, BlockResult, exclude_floating, merge_results
from ai_eda.design.rf.blocks.power import PACK_CUTOFF_V, PACK_MAX_V, PowerBlock, exclude, net, serving
from ai_eda.design.rf.blocks.ptt import TOT_RANGE_S, TX_TIMEOUT_KEY, PttBlock
from ai_eda.design.rf.blocks.rx_audio import RxAudioBlock
from ai_eda.design.rf.blocks.tx_audio import TxAudioBlock
from ai_eda.design.rf.family import BUILDS, refusal_question, unserved_message
from ai_eda.design.rf.models import MODEL_VERDICT
from ai_eda.design.rf.parts import part_line
from ai_eda.design.rf.profile import PROFILE, profile_choices, profile_keys
from ai_eda.design.templates import THEORY_CURVE_NOTE, _changes, _curve_figure, _log_grid, _lin_grid, _missing_inputs, _net_line, _refused

if TYPE_CHECKING:
    from ai_eda.design.board import BoardContext, SIDeclarations
    from ai_eda.report.figures import Figure

TEMPLATE_ID = "kr447_audio_ptt"
BUILD = "audio_ptt"
#: the pack voltages the board is designed for: above the low-pack inhibit's release point, at most a full 2S pack
V_IN_MIN = PACK_CUTOFF_V + 0.2
V_IN_MAX = PACK_MAX_V
#: the deviation / audio-bandwidth range the audio chain is designed for (Hz)
DEVIATION_RANGE = (500.0, 5000.0)
BANDWIDTH_RANGE = (2000.0, 4000.0)
#: why the stage-1 board allows 2 or 4 layers and defaults to 4
LAYER_REASON = "no RF on this board: 2 or 4 layers; 4 by default for one stack across the stages"
#: block id -> (x, y, w, h) mm of its floorplan region (origin top-left, Y down): 74 x 82 mm, the smallest arrangement of these five
#: rectangles found in which ``placement.rf_floorplan`` packs every part with and without the time-out (measured on the KiCad 10.0.6 footprints);
#: the TX audio region is 37 mm tall (36 mm no longer holds its parts with the through-hole microphone's 6.5 x 6.5 mm courtyard) and the PTT
#: region below it 45 mm (it still holds the time-out)
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "rx_audio": (0.0, 0.0, 42.0, 42.0),
    "tx_audio": (42.0, 0.0, 32.0, 37.0),
    "ptt": (42.0, 37.0, 32.0, 45.0),
    "power": (0.0, 42.0, 42.0, 30.0),
    "bench": (0.0, 72.0, 42.0, 10.0),
}
_REGION_WHAT = {
    "rx_audio": "RX audio, volume and squelch at the top left (packed from the corner: the squelch pot and the speaker jack land inside the board, not on an edge)",
    "tx_audio": "TX audio (microphone, limiter, splatter filter, integrator) at the top right",
    "ptt": "PTT sequencer, TX interlock and PA supply switch (and the time-out when built) at the bottom right (the PTT switch lands in the board's interior, not on the top edge)",
    "power": "pack connector, main switch, rail switches and regulators at the bottom left (the pack connector lands on the left edge, not the bottom edge)",
    "bench": "the bench interface headers J402-J404 along the bottom edge",
}


def stage1_blocks(*, tot: bool, pa_stand_in: bool = True) -> list[tuple[RFBlockBuilderBase, BlockPrefix]]:
    """The four stage-1 blocks with their prefixes (the composition's numbering); the transceiver composes the same builders."""
    return [
        (PowerBlock(), BlockPrefix(ref_base=100)),
        (PttBlock(pa_stand_in=pa_stand_in, tot=tot), BlockPrefix(ref_base=200)),
        (TxAudioBlock(), BlockPrefix(ref_base=300)),
        (RxAudioBlock(), BlockPrefix(ref_base=400)),
    ]


class BenchHeadersBlock(RFBlockBuilderBase):
    """J402-J404: the bench interface of the stage-1 board (what the RX back-end / TX exciter will drive or read)."""

    id = "bench"
    title = "bench interface headers"
    interface_nets = ("DISC_OUT", "RSSI", "PM_DRIVE", "PA_ON", "MUTE", "PA_PD")

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        members: dict[str, list[tuple[str, str]]] = {"GND": []}
        for ref, a, c, what in (("J2", "DISC_OUT", "RSSI", "the FM detector's audio and RSSI in (from a generator or the RX back-end)"),
                                ("J3", "PM_DRIVE", "PA_ON", "the integrated TX audio and the PA enable out (to the TX exciter)"),
                                ("J4", "MUTE", "PA_PD", "the squelch mute and the PA power-down out")):
            j = b.part("header_3", ref, "Conn_01x03", f"bench header: {what}", serving(ctx, "radio_build"))
            exclude(b, ref, "connector: the bench interface is not simulated")
            members.setdefault(a, []).extend(j.at("Pin_1"))
            members.setdefault(c, []).extend(j.at("Pin_2"))
            members["GND"].extend(j.at("Pin_3"))
        for name, pins in members.items():
            net(b, name, pins, f"{name} on the bench headers")
        b.result.chain = ["J2", "J3", "J4"]
        return b.done()


def _board_values(ctx: BlockContext) -> BlockResult:
    """The rows that belong to the board, not to a block: the KR 447 MHz profile placeholders and the floorplan regions."""
    b = BlockBuilder(ctx, "board", "board values")
    for ch, traced in profile_choices(ctx.template_id, ctx.confirmed):
        b.result.choices.append(ch)
        b.result.params[ch.key] = traced
    for bid, box in REGIONS.items():
        b.choice(f"floor.{bid}", list(box), "mm", f"floorplan region of the {bid} block (x, y, w, h mm from the top-left corner): {_REGION_WHAT[bid]}")
    return b.done()


class Kr447AudioPttTemplate(Template):
    """KR 447 MHz stage-1 bench board: power, PTT sequencer / interlock, TX audio processor, RX audio amplifier (module docstring)."""

    id = TEMPLATE_ID
    title = "KR447 audio / PTT bench board"
    triggers = ()
    needs = ("input_voltage", "modulation")
    serves = BUILDS[BUILD].serves
    layer_policy = LayerPolicy(allowed=(2, 4), default=4, reason=LAYER_REASON)
    plane_nets = ("GND", None)

    def triggered_by(self, ir: CircuitIR, inputs: dict[str, DesignInput]) -> bool:
        return read_radio_build(ir)[0] == BUILD

    def refusals(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str]) -> list[MissingInformation]:
        """The family's closed world (a requirement another build serves names that build) and the FM-only rule."""
        out: list[MissingInformation] = []
        modulation, _why = read_modulation(ir)
        if modulation is not None and modulation != "fm":
            why = f"modulation {modulation}: the KR 447 MHz licence-exempt class is FM (F3E) [UNVERIFIED: 「무선설비규칙」]; radio_build={BUILD} builds only an FM audio chain"
            out.append(MissingInformation(key="modulation", required=False, rationale=why,
                                          question=with_remedy(f"{why}; no template design was proposed. State modulation=FM or choose another design", unusable_remedy(ir, MODULATION_ALIASES)) + "."))
        unserved = unserved_requirements(ir, self)
        keys = [r.key for r in unserved]
        for r in unserved:
            canon = canonical_key(r.key) or r.key
            why = f"{r.id} ({requirement_text(r)}): {unserved_message(canon, BUILD)}"
            out.append(refusal_question(why, r.key, BUILD, keys))
        return out

    def _out_of_range(self, inputs: dict[str, DesignInput]) -> str | None:
        v = inputs.get("input_voltage")
        if v is not None and not V_IN_MIN <= v.traced.value <= V_IN_MAX:
            return (f"input_voltage {v.traced.value:.12g} V ({v.requirement.id}) is outside {V_IN_MIN:.12g}..{V_IN_MAX:.12g} V: the 2S Li-ion pack is used only "
                    f"above the confirmed cut-off {PACK_CUTOFF_V:.12g} V plus the low-pack inhibit's 0.2 V hysteresis (below its release point the transmitter "
                    f"is inhibited by design), and a full 2S pack is {PACK_MAX_V:.12g} V (the P-FET gates see the whole pack [UNVERIFIED: AOS AO3401A datasheet])")
        for key, (lo, hi), why in (("frequency_deviation", DEVIATION_RANGE, "the limiter / integrator chain's design range (the KR profile's maximum is a separate check)"),
                                   ("audio_bandwidth", BANDWIDTH_RANGE, "the splatter filter's and the RX low-pass's design range (a voice channel)"),
                                   (TX_TIMEOUT_KEY, TOT_RANGE_S, "the 4060 RC time-out's design range with a 100 nF timing capacitor")):
            inp = inputs.get(key)
            if inp is not None and not lo <= inp.traced.value <= hi:
                return f"{key} {inp.traced.value:.12g} {inp.traced.unit} ({inp.requirement.id}) is outside {lo:.12g}..{hi:.12g} {inp.traced.unit}: {why}"
        return None

    def build(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str], library: KicadLibrary, *, confirmed: bool) -> Plan:
        t = self.id
        plan = Plan(template=t, title=self.title)
        stated = {k: inputs[k] for k in ("input_voltage", "frequency_deviation", "audio_bandwidth", TX_TIMEOUT_KEY) if k in inputs}
        why = self._out_of_range(inputs)
        if why is not None:
            plan.inputs = stated
            return _refused(plan, why)
        present = present_keys(ir, inputs)
        if "input_voltage" not in present:
            _missing_inputs(plan, "The KR447 audio / PTT bench board template", ["input_voltage"], unusable, examples={"input_voltage": "7.4 V"}, ir=ir)
        modulation, mod_why = read_modulation(ir)
        if modulation is None:
            plan.questions.append(MissingInformation(key="modulation", required=mod_why is None, rationale="template input",
                                                     question=with_remedy(mod_why, unusable_remedy(ir, MODULATION_ALIASES)) or "The KR447 audio / PTT bench board template needs modulation: answer modulation=FM"))
            return _refused(plan, f"modulation missing{': ' + mod_why if mod_why else ''}")
        if plan.questions or plan.notes:
            return plan
        if modulation != "fm":
            return _refused(plan, f"modulation {modulation}: the KR 447 MHz class is FM [UNVERIFIED]")
        plan.inputs = stated
        ctx = BlockContext(ir=ir, library=library, template_id=t, confirmed=confirmed, inputs=inputs)
        try:
            built = [(blk, blk.build(ctx, prefix)) for blk, prefix in stage1_blocks(tot=TX_TIMEOUT_KEY in inputs)]
            built.append((BenchHeadersBlock(), BenchHeadersBlock().build(ctx, BlockPrefix(ref_base=400))))
            merged = merge_results([r for _, r in built] + [_board_values(ctx)], t)
        except TemplateRefusal as e:
            return _refused(plan, str(e))
        components, report = exclude_floating(merged.components, merged.nets, merged.stimuli, t)
        bad = [f for f in report.unresolved if not f.ref.startswith("RV")]
        if bad:
            return _refused(plan, "a multi-terminal part has a node nothing else simulated touches: " + "; ".join(f.reason for f in bad))
        blocks = [r for _, r in built]
        plan.choices = list(merged.choices)
        plan.computed = list(merged.computed)
        plan.parts = [part_line(merged.placed[c.ref]) for c in components]
        plan.nets = [_net_line(n) for n in merged.nets]
        sim = SimulationSetup(stimuli=list(merged.stimuli), analyses=list(merged.analyses), expectations=list(merged.expectations))
        plan.simulation = self._simulation_lines(sim, components, report)
        rf = RFDesign(
            blocks=[RFBlock(id=r.block_id, title=r.title, refs=[c.ref for c in r.components], chain=list(r.chain), region=self._region(merged, r.block_id), ports=list(r.ports))
                    for r in blocks],
            lab_items=list(merged.lab_items), rails=list(merged.rails), model_values=list(merged.model_keys), profile_keys=profile_keys(),
        )
        topology = Topology(
            name="KR447 audio / PTT bench board", domains=[CircuitDomain.POWER, CircuitDomain.ANALOG, CircuitDomain.DIGITAL],
            rationale=("stage 1 of the KR 447 MHz FM walkie-talkie family: 2S pack -> main P-FET switch -> PTT-switched RX / TX rails and regulators; "
                       "PTT sequencer with settle delay, low-pack TX inhibit and AND interlock driving the PA supply switch; TX audio: MAX9814 -> "
                       "high-pass + pre-emphasis -> diode limiter -> 4th-order Butterworth splatter filter -> deviation trim -> lossy integrator -> PM_DRIVE; "
                       "RX audio: de-emphasis + 300-3000 Hz band-pass -> volume -> LM386; squelch comparator. No RF on this board"),
            provenance=ctx.provenance("selected by radio_build = audio_ptt"),
            blocks=[Block(id=r.block_id, function=r.title, domain=CircuitDomain.POWER if r.block_id == "power" else CircuitDomain.ANALOG,
                          component_refs=[c.ref for c in r.components], provenance=ctx.provenance(f"block {r.block_id}")) for r in blocks],
        )
        constraints = self._constraints(ctx, merged)
        plan.changes = _changes(t, self.title, topology, components, merged.nets, dict(merged.params), sim, constraints)
        plan.changes.append(DesignChange(description="RF design: blocks, rails, lab items, model values, profile", target="rf", operation="set", payload=rf,
                                         rationale=f"template {t}: the family's blocks and their unverified values"))
        return plan

    # --- build steps ------------------------------------------------------------------

    @staticmethod
    def _region(merged: BlockResult, bid: str) -> RFRegion | None:
        t = merged.params.get(f"floor.{bid}")
        if t is None:
            return None
        x, y, w, h = (float(v) for v in t.value)
        return RFRegion(**{k: Traced(value=v, unit="mm", provenance=t.provenance) for k, v in (("x", x), ("y", y), ("w", w), ("h", h))})

    @staticmethod
    def _simulation_lines(sim: SimulationSetup, components, report) -> list[str]:
        lines = [f"stimulus {s.id} ({s.kind.value}) on {s.net}: {s.provenance.note}" for s in sim.stimuli]
        for a in sim.analyses:
            params = ", ".join(f"{k} {v.value!r}" for k, v in a.params.items())
            lines.append(f"analysis {a.id} ({a.kind.value}{': ' + params if params else ''})")
        for e in sim.expectations:
            rule = f"{e.bound} {e.nominal.value:.6g}" if e.bound else f"{e.nominal.value:.6g} +/- " + (f"{e.tol_abs.value:.3g}" if e.tol_abs else f"{e.tol_rel.value:.0%}")  # type: ignore[union-attr]
            at = f" at {e.at.value:.6g}" if e.at is not None else ""
            ref = f" re {e.reference_vector}" if e.reference_vector else ""
            lines.append(f"expectation {e.id}: {e.vector}{ref} {e.reduce.value}{at} {rule} {e.nominal.unit or ''} ({e.provenance.note})")
        excluded = [c.ref for c in components if c.spice is not None and c.spice.exclude]
        lines.append(f"excluded from the netlist ({len(excluded)}): {', '.join(excluded)} (reasons in each part's SPICE binding; "
                     f"{len(report.excluded)} of them dead branches found by exclude_floating)")
        lines += [f"reported, kept: {f.reason}" for f in report.unresolved]
        lines.append(f"every PASS here is a {MODEL_VERDICT}: no IC is simulated, no RF is on this board")
        return lines

    @staticmethod
    def _constraints(ctx: BlockContext, merged: BlockResult) -> list[Constraint]:
        s = ctx.provenance
        return [
            Constraint(id="c.kr447.no_transmission", kind=ConstraintKind.REGULATORY, target="*", provenance=s("stage-1 bench board"),
                       description=("stage 1 of the KR 447 MHz family: no RF on this board. Every kr447.* number is an UNVERIFIED placeholder; a radio built from "
                                    "the family needs KC conformity assessment (전파법 제58조의2) before any transmission, a self-built unit included")),
            Constraint(id="c.kr447.logic_supply", kind=ConstraintKind.ELECTRICAL, target="TX_3V3", provenance=s("absolute maximum supply of the logic"),
                       description=("U201 / U204 (LMV331), U202 (74LVC1G08) and U203 (4060) run from TX_3V3 and U403 from RX_3V3, never from V_SYS / V_TX "
                                    "(up to 8.4 V; their absolute maximum is about 5.5 V [UNVERIFIED: TI LMV331, 74LVC1G08 datasheets])")),
            Constraint(id="c.kr447.pack", kind=ConstraintKind.ELECTRICAL, target="J101", provenance=s("pack condition"),
                       parameters={"pack_cutoff_v": merged.params["power.pack_cutoff_v"]},
                       description=("J101 takes a protected 2S Li-ion pack charged externally (no on-board charger); it is used only above power.pack_cutoff_v "
                                    "(the low-pack TX inhibit enforces it for TX; its own protection cut-off is lower [UNVERIFIED: pack datasheet])")),
            Constraint(id="c.kr447.pa_pd_polarity", kind=ConstraintKind.ELECTRICAL, target="PA_PD", provenance=s("PA power-down polarity"),
                       description=("PA_PD high = PA powered down (pa.power_down_polarity, from the library pin name POWER_DOWN "
                                    "[UNVERIFIED: NXP MMZ09332B datasheet]); check the polarity before the TX exciter uses this line")),
        ]

    # --- signal integrity ----------------------------------------------------------------

    #: the pack / TX path current the POWER_PA class is sized for, and the nets it holds
    POWER_PA_CURRENT_A = 0.6
    POWER_PA_TEMP_RISE_C = 10.0
    POWER_PA_NETS: tuple[str, ...] = ("VBAT", "VBAT_F", "V_SYS", "V_TX", "TX_5V", "PA_5V")

    def si_declarations(self, ir: CircuitIR, ctx: BoardContext) -> SIDeclarations:
        """``POWER_PA``: the pack and TX supply path at least the IPC-2221 width for 0.6 A (the transceiver's TX current with margin)."""
        from ai_eda.design.board import SIDeclarations
        from ai_eda.tools.calc.basic import ipc2221_width_for_current

        out = SIDeclarations()
        i = ctx.choice("si.power_pa_current", self.POWER_PA_CURRENT_A, "A", (
            "current the pack / TX supply path's minimum width is sized for: 0.6 A, above the transceiver's 441 mA TX budget "
            "[UNVERIFIED: the budget is an estimate]"), param=True)
        dt = ctx.choice("si.power_pa_temp_rise", self.POWER_PA_TEMP_RISE_C, "degC", "temperature rise the POWER_PA class's IPC-2221 width is sized for", param=True)
        if ctx.stackup is not None:
            layer = ctx.stackup.copper_layer("F.Cu")
            assert layer is not None
            t_id = "pcb.stackup.copper[F.Cu].thickness_um"
            w = ctx.computed_param("si.w_power_pa", ipc2221_width_for_current(i, dt, layer.thickness_um, ("si.power_pa_current", "si.power_pa_temp_rise", t_id)))
            out.classes.append(NetClass(name="POWER_PA", nets=list(self.POWER_PA_NETS), min_width_mm=w, power_current_a=i, power_temp_rise_c=dt,
                                        description="the pack and TX supply path: at least the IPC-2221 width for 0.6 A",
                                        provenance=ctx.structural("pack / TX supply path minimum width")))
        return out

    # --- report hooks (views: numbers from ir.parameters, recomputed only for display) ----

    def _p(self, ir: CircuitIR, key: str) -> float | None:
        return parameter_value(ir, key)

    def theory(self, ir: CircuitIR) -> list[TheorySection]:
        p = lambda key: self._p(ir, key)  # noqa: E731
        q = lambda key, unit: quantity(p(key), unit)  # noqa: E731
        v_in, v_th, v_fin, tau = p("power.v_in"), p("ptt.v_th"), p("ptt.v_dly_final"), p("ptt.tau_dly")
        t_d = None
        if None not in (v_th, v_fin, tau) and v_fin > v_th > 0:  # type: ignore[operator]
            t_d = tau * math.log(v_fin / (v_fin - v_th))  # type: ignore[operator]
        r6, r7, r11, v_ref, v_hi = p("ptt.r_uv_top"), p("ptt.r_uv_bottom"), p("ptt.r_uv_hyst"), p("ptt.v_uv_ref"), p("power.tx_3v3")
        v_fall = v_rise = None
        if None not in (r6, r7, r11, v_ref, v_hi):
            g = 1.0 / r6 + 1.0 / r7 + 1.0 / r11  # type: ignore[operator]
            v_rise = v_ref * g * r6  # type: ignore[operator]
            v_fall = (v_ref * g - v_hi / r11) * r6  # type: ignore[operator]
        hd = []
        for rail, reg, key in (("RX_5V", "U101", "rx_5v"), ("TX_5V", "U102", "tx_5v")):
            vo, i_max, drop, rp, cut = p(f"power.{key}"), p(f"power.{key}.i_max"), p(f"power.{key}.dropout"), p("power.path_r"), p("power.pack_cutoff_v")
            head = None if None in (vo, i_max, drop, rp, cut) else cut - i_max * rp - drop - vo  # type: ignore[operator]
            hd.append(f"| {rail} ({reg}) | {quantity(cut, 'V')} − {quantity(i_max, 'A')}·{quantity(rp, 'ohm')} − {quantity(drop, 'V')} − {quantity(vo, 'V')} | {quantity(head, 'V')} |")
        profile_rows = "\n".join(f"| `{e.key}` | {e.value if isinstance(e.value, str) else quantity(float(e.value), e.unit)} | {e.document} |" for e in PROFILE)
        n_mult, k_pm, v_lim, tau_i, df_d = p("rf.n_mult"), p("model.k_pm"), p("tx.v_lim"), p("tx.tau_i"), p("tx.delta_f_design")
        dev, v_max, hr = p("tx.frequency_deviation"), p("tx.v_splat_max"), p("tx.deviation_headroom")
        pm_300 = None if None in (v_lim, tau_i) else v_lim / (2 * math.pi * 300.0 * tau_i)  # type: ignore[operator]
        tot = ""
        if p("ptt.tot_period") is not None:
            tot = (f"\n\n**송신 시간 제한(TOT)** tx_timeout 이 명시되어 4060(U203) RC 발진기와 14단 카운터를 넣었습니다. 시간 제한은 "
                   f"T = 2^13 · k_RC · R_t · C_t = 8192 · {number(p('ptt.tot_k_rc'))} · {q('ptt.tot_r', 'ohm')} · {q('ptt.tot_c', 'F')} = **{q('ptt.tot_period', 's')}** "
                   f"(`calc.rf.tot.period`, 명시값 {q('ptt.tx_timeout', 's')} 이하가 되도록 E24 에서 R_t 를 고름). Q14 가 올라가면 Q204 가 DLY 를 잡아 PA 가 꺼지고 D203 이 발진을 멈춰 "
                   "PTT 를 놓을 때까지 유지됩니다. k_RC 는 4060 데이터시트 값으로 검증되지 않았고(검증되지 않음), 카운터는 시뮬레이션하지 않으므로 실제 시간은 실험 항목 `rf.lab.tot` 입니다.")
        return [
            TheorySection("개요: 1단계 오디오 / PTT 시험 기판", (
                "KR 447 MHz 면허 불요 FM 무전기 계열의 1단계 기판입니다. RF 는 없고, 무전기의 **펌웨어 없는 제어와 오디오**를 지금 있는 도구(op, ac, tran 해석과 dB 축약, 단측 한계)로 "
                "검증합니다. 블록은 전원(1xx), PTT 순서 제어·송신 인터록(2xx), 송신 오디오(3xx), 수신 오디오(4xx), 시험용 헤더 J402–J404 입니다.\n\n"
                "**모든 KR 규제 수치는 검증되지 않은 자리표시값입니다.** 「신고하지 아니하고 개설할 수 있는 무선국용 무선기기」와 「무선설비규칙」을 공식 원문으로 확인하기 전까지 "
                "`kr447.*` 값은 사용자가 확인한 선택값일 뿐 근거 있는 사실이 아니며 `rf.regulatory_profile` 은 PASS 가 되지 않습니다. 이 계열로 만든 무전기는 자작품이라도 "
                "송신 전에 KC 적합성평가(전파법 제58조의2)를 받아야 합니다. 이 기판은 송신하지 않습니다.\n\n"
                "배치(설계 §2.1 과 다름): 설계는 커넥터를 아래쪽 가장자리에, PTT 와 가변저항을 위쪽 가장자리에 두라고 하지만, 영역 배치기는 각 블록의 부품을 "
                "영역 왼쪽 위부터 차례로 채우므로 사용자가 만지는 부품이 가장자리에 오지 않습니다. 74 × 82 mm 기판에서(두 빌드 같음, 가장자리 여백 2 mm) "
                "RV401(음량)과 MK301 은 채우는 순서 덕에 위쪽 가장자리에 있지만, RV402(스퀠치)는 위쪽 가장자리에서 16.3 mm 안쪽, SW201(PTT)은 기판 안쪽"
                "(위쪽에서 37.5 mm), J101(팩)은 왼쪽 가장자리(아래쪽에서 34 mm 위), J401(스피커)은 기판 안쪽(아래쪽에서 60.2 mm 위)에 놓입니다. 아래쪽 "
                "가장자리에는 시험용 헤더 J402–J404 만 있습니다. 이 부품들의 가장자리 배치는 KiCad 에서 손으로 옮기는 일로 남겨 둡니다.\n\n"
                "| 규제 프로파일 키 | 값 (검증되지 않음) | 확인할 문서 |\n|---|---|---|\n" + profile_rows
            )),
            TheorySection("전원과 PTT 레일 전환", (
                f"2S 리튬이온 팩(명시 전압 {quantity(v_in, 'V')}) → F101 → Q105(P-FET 주 스위치, 게이트를 SW101 이 GND 로 당김) → V_SYS. "
                "PTT 를 누르면 PTT_N 이 낮아져 Q103 이 꺼지고 PTT_ACTIVE 가 V_SYS 로 올라갑니다. PTT_ACTIVE 는 RX 스위치 Q101 의 게이트이므로 RX 레일이 꺼지고, "
                "Q104 를 통해 Q102 의 게이트를 끌어내려 TX 레일(V_TX)이 켜집니다. 놓으면 반대입니다.\n\n"
                f"레귤레이터: U101 LP38693DT-5.0 (V_RX → RX_5V, 결정 3A), U103/U104 LP5907-3.3, U102 LM1117-5.0 (V_TX → TX_5V). 팩 하한 선택값 power.pack_cutoff_v = {q('power.pack_cutoff_v', 'V')} "
                "에서의 헤드룸(`calc.regulator.headroom`, 모든 전류·드롭아웃은 검증되지 않은 선택값):\n\n"
                "| 레일 | V_in,min − I·R_path − V_dropout − V_out | 헤드룸 |\n|---|---|---|\n" + "\n".join(hd) + "\n\n"
                "설계 덱에서 레귤레이터는 모델이 없으므로 입력을 저항(`model.ldo_rx.r_in`, `model.ldo_tx.r_in`)으로, 출력 레일을 이상 전원으로 둡니다. 그래서 레일 순서 검사"
                "(`tx_rail_on`, `rx_rail_off_tx`, `tx_rail_off_rx`)는 스위치와 인버터를 범용 모델(`model.pmos`, `model.npn`)로 본 원리 검증입니다."
            )),
            TheorySection("송신 대기(settle) 지연과 인터록", (
                "PA 는 PTT 가 눌렸고 **그리고** 지연 비교기가 동작했을 때만 켜집니다(기본값은 송신하지 않음). 지연은 V_TX 에서 R201, GND 로 R202 인 테브난 RC 로 C201 을 충전합니다:\n\n"
                "    V_DLY(t) = V_final · (1 − e^(−t/τ)),  V_final = V_TX · R202/(R201 + R202),  τ = (R201 ∥ R202) · C201\n\n"
                f"이 설계에서 V_final ≈ {q('ptt.v_dly_final', 'V')} (팩 전압 기준, `calc.divider.v_out`), τ = {q('ptt.tau_dly', 's')} (`calc.rc.tau`), 문턱 V_th = TX_3V3 · R204/(R203 + R204) = "
                f"{q('ptt.v_th', 'V')} (`calc.divider.v_out`). 교차 시각 t_d = τ · ln(V_final/(V_final − V_th)) = {quantity(t_d, 's')} (표시용 계산).\n\n"
                f"- t_settle = {q('ptt.t_settle', 's')} 에서 V_DLY = {q('ptt.v_dly_settle', 'V')} < V_th 이어야 함 (`pa_held_off_while_settling`)\n"
                f"- t_delay_max = {q('ptt.t_delay_max', 's')} 에서 V_DLY = {q('ptt.v_dly_enable', 'V')} > V_th 이어야 함 (`pa_enabled_after_settle`)\n"
                "- PTT 를 놓으면 Q103 이 PTT_ACTIVE 를 끌어내리고 D201 이 C201 을 곧바로 방전 (`pa_off_fast`, 해제 1 ms 뒤)\n\n"
                "AND 게이트 U202 는 PTT_ACTIVE 를 R213/R214 로 절반으로 나눈 PTT_LOGIC(입력 허용 5.5 V, 검증되지 않음)과 DLY_OK 를 곱해 PA_ON 을 만듭니다. 비교기·AND 게이트는 모델이 없어 "
                "제외되고, 설계 덱의 PA_ON 은 인터록이 늦어도 켜야 할 시각부터 해제까지 높은 PWL 입니다(모델링 선택값). 그 PA_ON 으로 PA 전원 스위치 Q201 이 켜지고 "
                "(`pa_supply_on`) 해제 1 ms 뒤 PA_5V 가 0.5 V 아래로 떨어지는지(`pa_supply_off_first`, 이 기판에는 PA 가 없어 R219 가 부하를 대신함) 확인합니다.\n\n"
                "지연 비교기 U201 에는 설계서가 말한 히스테리시스를 넣지 않았습니다: 입력이 단조 증가하는 RC 램프이고(D201 이 해제 때 리셋) 궤환 저항은 타이밍 노드를 부하합니다."
            )),
            TheorySection("저전압 송신 금지 (결정 3A)", (
                f"팩 하한 {q('power.pack_cutoff_v', 'V')} 를 하드웨어로 지킵니다. U204 가 V_TX 분배 UV_SENSE(R206/R207)와 TX_3V3 분배 UV_REF(R208/R209, {q('ptt.v_uv_ref', 'V')}, `calc.divider.v_out`)를 "
                "비교하고, 오픈 컬렉터 출력 LOWBAT_N 이 R211 로 되먹임(히스테리시스)되며, 낮아지면 D202 로 DLY 를 잡아 DLY_OK 가 올라가지 못합니다.\n\n"
                "    G = 1/R206 + 1/R207 + 1/R211\n"
                "    복귀(상승) V_TX,rise = V_REF · G · R206\n"
                "    동작(하강) V_TX,fall = (V_REF · G − V_TX3V3/R211) · R206,  폭 = V_TX3V3 · R206/R211\n\n"
                f"이 설계의 저항 선택값으로 V_TX,fall = {quantity(v_fall, 'V')}, V_TX,rise = {quantity(v_rise, 'V')} (표시용 계산, 선택값 power.uvlo_hyst_v = {q('power.uvlo_hyst_v', 'V')}). "
                "비교기는 시뮬레이션하지 않으므로 설계 덱은 명시 팩 전압에서 UV_SENSE 가 UV_REF 위인지만 봅니다(`uvlo_sense_released`). 실제 동작·복귀점은 실험 항목 `rf.lab.uvlo` 입니다." + tot
            )),
            TheorySection("송신 오디오: 프리엠퍼시스, 리미터, 스플래터 필터", (
                f"MAX9814(U301, AGC; 스트랩은 검증되지 않은 선택값) → U302: C307/R306 고역통과 {q('tx.hpf_corner', 'Hz')} 와 프리엠퍼시스 영점 1 + s·R307·C308 "
                f"(τ = {q('tx.preemph_tau', 's')}, f_e = {q('tx.f_preemph', 'Hz')}). 단 전체의 공칭 응답은\n\n"
                "    A(f) = −10·log10(1 + (f_hp/f)²) + 10·log10(1 + (f/f_e)²)   [dB]   (`calc.audio.highpass1.db_at` + `calc.audio.emphasis.db_at`)\n\n"
                f"리미터: C310/R308 을 거쳐 역병렬 다이오드 D301/D302 가 CLIP 을 ±V_D = {q('tx.clip_vd', 'V')} 로 자르고, U305 가 받아 LIM_OUT 으로 냅니다.\n\n"
                f"스플래터 필터(결정 4B): 4차 버터워스, 두 단 단위이득 Sallen-Key (U303 Q = {number(p('splat.q1'), 5)}, U306 Q = {number(p('splat.q2'), 5)}, "
                "`calc.audio.butterworth.q`), R = 10 kΩ, C1 = 2Q/(ω_c R), C2 = 1/(2Q ω_c R) (`calc.audio.sallen_key.*`):\n\n"
                f"| 단 | C1 | C2 |\n|---|---|---|\n| U303 | {q('splat.c1_1', 'F')} | {q('splat.c2_1', 'F')} |\n| U306 | {q('splat.c1_2', 'F')} | {q('splat.c2_2', 'F')} |\n\n"
                f"감쇠 A(f) = 10·log10(1 + (f/f_c)^8): 코너 {q('audio.bandwidth', 'Hz')} 에서 {q('splat.att_bw', 'dB')}, 두 배 주파수에서 ≥ {q('tx.splat_min_2bw', 'dB')}, 채널 간격 "
                f"{q('kr447.channel_raster', 'Hz')} 에서 ≥ {q('tx.splat_min_raster', 'dB')} 를 요구합니다. 1 kHz 톤을 +20 dB 과구동해 잘린 파형의 고조파는 "
                f"h3 ≤ {q('tx.splat_h3_max', 'dBc')}, h5 ≤ {q('tx.splat_h5_max', 'dBc')}, h7 ≤ {q('tx.splat_h7_max', 'dBc')} (h3 는 코너에 있어 대역 내 왜곡이지 스플래터가 아님)."
            )),
            TheorySection("적분기와 주파수 편이 (간접 FM)", (
                "위상 변조기 앞에서 제한된 오디오를 적분하면 위상 변조가 주파수 변조가 됩니다(Armstrong). 체배수 N, 변조기 기울기 K_pm, PM_DRIVE→바랙터 계수 a 에서\n\n"
                "    Δf = N · K_pm · a · V_max / (2π · τ_i)        … (편이 식, `rf.deviation`)\n\n"
                f"이 기판: N = {number(p('rf.n_mult'))} (결정 1B: 3·2·2), K_pm = model.k_pm = {q('model.k_pm', 'rad/V')} (두 탱크 합의 **모델 값**, 검증되지 않음), "
                f"설계 편이 Δf_design = 편이 요구 {quantity(dev, 'Hz')} / {number(hr)} = {quantity(df_d, 'Hz')} (`calc.clock.divided`), "
                f"적분기 입력의 설계 레벨 V_lim = V_D × 트림 위치 = {quantity(v_lim, 'V')} (`calc.rf.limiter.level`).\n\n"
                f"    τ_i = N · K_pm · V_lim / (2π · Δf_design) = {quantity(tau_i, 's')}   (`calc.rf.fm.pm_integrator_tau`)\n"
                f"    R316 = τ_i / C322 = {q('tx.r_int', 'ohm')}   (`calc.rc.r_for_cutoff` at 1/(2π τ_i))\n"
                f"    V_max 한계 = 2π · τ_i · Δf / (N · K_pm) = {quantity(v_max, 'V')}   (`calc.rf.fm.pm_drive_limit`, `pm_drive_peak` 의 상한)\n\n"
                f"여유 계수 {number(hr)} 는 설계서의 0.9(=1.11) 대신 쓴 값입니다. ngspice-42 에서 +20 dB 과구동 시 잘린 1 kHz 파형이 4차 필터에서 오버슈트해 INT_IN 피크가 V_lim 의 약 1.25 배가 되므로 "
                "1.11 로는 편이 요구를 넘습니다. 300 Hz 에서 PM_DRIVE 진폭 ≈ V_lim/(2π·300·τ_i) = " + quantity(pm_300, 'V') + " (a = 1 일 때 두 탱크 바랙터 스윙 약 0.56 V 에 해당).\n\n"
                "K_pm 이 모델 상수이므로 이 기판의 `rf.deviation` 은 NOT_VERIFIED 입니다. 실제 편이는 송신 여진기 단계의 실험 항목입니다."
            )),
            TheorySection("수신 오디오와 스켈치", (
                "DISC_OUT → U401 반전단: 입력 가지 C403 + R403 가 고역통과, 궤환 R404 ∥ C404 가 디엠퍼시스 극점(R403 = R404 = R):\n\n"
                "    H(s) = −(s·R·C403 / (1 + s·R·C403)) · 1/(1 + s·R·C404)   (정확한 곱)\n\n"
                f"R = τ_d/C404 = {q('rx.r_stage', 'ohm')} (τ_d = {q('rx.deemph_tau', 's')}), 고역 코너 f_hp = 1/(2π·R·C403) = {q('rx.f_hp', 'Hz')}, "
                f"그 뒤 R405/C405 1차 저역통과 {q('audio.bandwidth', 'Hz')} (R405 = {q('rx.r_lp', 'ohm')}) → U404 → VOL_IN. 공칭 응답(dB)은 세 1차 항의 합(`calc.rf.db_sum`):\n\n"
                f"| 주파수 | 공칭 | \n|---|---|\n| {q('audio.f_low', 'Hz')} | {q('rx.hpde_low', 'dB')} + {q('rx.lp_low', 'dB')} |\n| {q('audio.f_ref', 'Hz')} | {q('rx.hpde_ref', 'dB')} + {q('rx.lp_ref', 'dB')} |\n"
                f"| {q('audio.bandwidth', 'Hz')} | {q('rx.hpde_bw', 'dB')} + {q('rx.lp_bw', 'dB')} |\n\n"
                f"스켈치: U403 이 RSSI 와 RV402 와이퍼(SQ_SET)를 비교, SQ_REF = RX_3V3 · RV402/(R410 + RV402) = {q('rx.v_sq_ref', 'V')} (`calc.divider.v_out`, `sq_threshold`). "
                "RSSI 가 문턱 아래면 MUTE 가 높아짐(뮤트, SA605 의 MUTE_INPUT 극성은 검증되지 않음). LM386 과 비교기는 모델이 없어 구조만 검증합니다."
            )),
            TheorySection("모델 값과 검증의 한계", (
                "이 기판에서 PASS 는 **확인된 모델 값 아래의 회로망·원리 판정**입니다(실제 부품의 측정이 아님). op-amp 는 A0 = 10^5, GBW = 1 MHz 의 단극 매크로(`model.opamp`, 레일 제한 없음), "
                "다이오드·NPN·PMOS 는 ngspice 기본/레벨 1 범용 카드, 전위차계는 와이퍼 위치 `model.pot.position` 의 두 저항입니다. `rf.model_grounding` 은 이 값들이 근거를 얻을 때까지 "
                "NOT_VERIFIED 이며, 설계 덱의 IC(MAX9814, LM386, LMV331, 74LVC1G08, 4060, 레귤레이터)는 하나도 시뮬레이션하지 않습니다.\n\n"
                "실험 항목(`rf.lab.*`): 마이크 AGC 레벨, 스플래터 스펙트럼, 레일 순서, 저전압 금지의 실제 동작점, 스피커 출력, 스켈치, 레일 전류."
            )),
        ]

    def theory_figures(self, ir: CircuitIR) -> list[Figure]:
        """(a) the TX high-pass x pre-emphasis response and the splatter filter's attenuation; (b) the RX chain's response; (c) the settle delay V_DLY(t)."""
        p = lambda key: self._p(ir, key)  # noqa: E731
        out: list[Figure] = []
        f_hp, f_e, bw = p("tx.hpf_corner"), p("tx.f_preemph"), p("audio.bandwidth")
        if None not in (f_hp, f_e, bw):
            xs = _log_grid(100.0, 20000.0, 200)
            pre = [-10 * math.log10(1 + (f_hp / f) ** 2) + 10 * math.log10(1 + (f / f_e) ** 2) for f in xs]  # type: ignore[operator]
            spl = [-10 * math.log10(1 + (f / bw) ** 8) for f in xs]  # type: ignore[operator]
            caption = (f"송신 오디오의 공칭 응답: 고역통과 {quantity(f_hp, 'Hz')} × 프리엠퍼시스 (f_e = {quantity(f_e, 'Hz')}) 와 4차 버터워스 스플래터 필터 −10·log10(1 + (f/f_c)^8), "
                       f"f_c = {quantity(bw, 'Hz')}. {THEORY_CURVE_NOTE}")
            out.append(_curve_figure("theory_tx_audio", "송신 오디오 응답 (프리엠퍼시스, 스플래터 필터)", caption, [("프리엠퍼시스 단", xs, pre), ("스플래터 필터", xs, spl)],
                                     x_label="주파수 f (Hz)", y_label="이득 (dB)", log_x=True, bands=[("x", bw, bw, f"f_c = {quantity(bw, 'Hz')}")]))
        f_rhp, f_de = p("rx.f_hp"), p("rx.f_deemph")
        if None not in (f_rhp, f_de, bw):
            xs = _log_grid(100.0, 20000.0, 200)
            ys = [-10 * math.log10(1 + (f_rhp / f) ** 2) - 10 * math.log10(1 + (f / f_de) ** 2) - 10 * math.log10(1 + (f / bw) ** 2) for f in xs]  # type: ignore[operator]
            caption = (f"수신 오디오 DISC_OUT → VOL_IN 의 공칭 응답: 고역통과 {quantity(f_rhp, 'Hz')} × 디엠퍼시스 {quantity(f_de, 'Hz')} × 저역통과 {quantity(bw, 'Hz')} (세 1차 항의 곱). "
                       f"{THEORY_CURVE_NOTE}")
            out.append(_curve_figure("theory_rx_audio", "수신 오디오 응답", caption, [("DISC_OUT → VOL_IN", xs, ys)], x_label="주파수 f (Hz)", y_label="이득 (dB)", log_x=True))
        v_fin, tau, v_th, t_s, t_m = p("ptt.v_dly_final"), p("ptt.tau_dly"), p("ptt.v_th"), p("ptt.t_settle"), p("ptt.t_delay_max")
        if None not in (v_fin, tau, v_th, t_s, t_m) and tau > 0:  # type: ignore[operator]
            xs = _lin_grid(0.0, 3.0 * tau, 300)  # type: ignore[operator]
            ys = [v_fin * (1 - math.exp(-x / tau)) for x in xs]  # type: ignore[operator]
            caption = (f"PTT 를 누른 뒤의 지연 노드 V_DLY(t) = V_final·(1 − e^(−t/τ)), V_final = {quantity(v_fin, 'V')}, τ = {quantity(tau, 's')}; 안내선 = 문턱 {quantity(v_th, 'V')}; "
                       f"점 = t_settle {quantity(t_s, 's')} 와 t_delay_max {quantity(t_m, 's')}. {THEORY_CURVE_NOTE}")
            out.append(_curve_figure("theory_settle_delay", "송신 대기 지연 V_DLY(t)", caption, [("V_DLY(t)", xs, ys)], x_label="PTT 를 누른 뒤 시간 t (s)", y_label="V_DLY (V)",
                                     bands=[("y", v_th, v_th, f"V_th = {quantity(v_th, 'V')}")],
                                     markers=[(t_s, v_fin * (1 - math.exp(-t_s / tau)), "t_settle"), (t_m, v_fin * (1 - math.exp(-t_m / tau)), "t_delay_max")]))  # type: ignore[operator]
        return out

    def theory_figure_vectors(self) -> tuple[str, ...]:
        return ("v(DLY)", "v(V_TX)", "v(V_RX)", "v(PA_5V)")

    def part_notes(self, ir: CircuitIR) -> dict[str, PartNote]:
        return stage1_part_notes(ir)


#: the stage-1 blocks whose parts :func:`stage1_part_notes` describes; every board that composes them keeps their numbering
#: (power 1xx, ptt 2xx, tx_audio 3xx, rx_audio 4xx), so the notes are keyed by reference
STAGE1_BLOCK_IDS: tuple[str, ...] = ("power", "ptt", "tx_audio", "rx_audio")


def stage1_refs(ir: CircuitIR) -> set[str]:
    """The references of the stage-1 blocks (:data:`STAGE1_BLOCK_IDS`) on ``ir``'s board, from ``ir.rf.blocks``."""
    if ir.rf is None:
        return set()
    return {r for blk in ir.rf.blocks if blk.id in STAGE1_BLOCK_IDS for r in blk.refs}


def stage1_part_notes(ir: CircuitIR, refs: set[str] | None = None) -> dict[str, PartNote]:
    """The Korean part notes of the stage-1 blocks' parts (all of ``ir``'s parts when ``refs`` is ``None``); the later stage boards reuse them."""
    notes: dict[str, PartNote] = {}
    for c in ir.components:
        if refs is not None and c.ref not in refs:
            continue
        note = _PART_NOTES.get(c.ref)
        if c.ref == "Q101" and "always on" in c.description:  # a receive-only board: no PTT sequencing, the gate is tied to GND
            note = _ic("Q101: RX 레일 P-FET 스위치 (수신 전용 보드: 항상 켜짐)", "AO3401A; 송신부가 없어 게이트를 GND 에 묶음(PTT 순서 제어 없음).",
                       ["P 채널, 문턱 |V_GS(th)| ≤ 1.3 V", "SOT-23 G/S/D 핀"], [])
        if note is not None:
            notes[c.ref] = note
            continue
        lib = f"{c.symbol.library}:{c.symbol.name}" if c.symbol is not None else NO_RECORD
        fp = f"{c.footprint.library}:{c.footprint.name}" if c.footprint is not None else NO_RECORD
        if c.symbol is not None and c.symbol.name in ("R", "C", "L", "C_Polarized", "Fuse"):
            kind = {"R": "저항", "C": "커패시터", "L": "인덕터", "C_Polarized": "전해 커패시터", "Fuse": "퓨즈"}[c.symbol.name]
            notes[c.ref] = PartNote(
                role=f"{c.ref}: {c.description}",
                why=f"{kind} {c.value} ({lib}, {fp}). 값은 계산기 출력 또는 사용자가 확인한 선택값이며 E 계열 반올림은 하지 않았음(부품 값 표기는 유효숫자 5자리).",
                criteria=[f"값 {c.value}", "정격 전압 ≥ 2 × 걸리는 전압 (최대 팩 전압 8.4 V 인 노드는 16 V 이상)", "같은 풋프린트 크기"],
                substitutes=[unverified(f"같은 값의 다른 제조사 {kind} (같은 풋프린트)")],
            )
        else:
            notes[c.ref] = PartNote(role=f"{c.ref}: {c.description}", why=f"{lib} / {fp} (라이브러리에서 핀 이름으로 배선).",
                                    criteria=["핀 배열과 정격을 데이터시트로 확인"], substitutes=[])
    return notes


def _ic(role: str, why: str, criteria: list[str], subs: list[str]) -> PartNote:
    return PartNote(role=role, why=why, criteria=criteria, substitutes=subs)


_PART_NOTES: dict[str, PartNote] = {
    "U101": _ic("U101: RX_5V 레귤레이터 (V_RX → RX_5V)",
                "결정 3A: LP2985-5.0(150 mA, SOT-23-5) 대신 TO-252-2 의 LP38693DT-5.0. 라이브러리 설명의 '500-mA' 로 골랐고 RX_5V 예산 86–161 mA(추정, 검증되지 않음)보다 큼. "
                "핀 순서 1 OUT / 2 GND / 3 IN 은 라이브러리를 따름(같은 TO-252-2 대안 BD50FC0FP·LF50_TO252 는 반대) - BOM 전에 TI 핀 표 확인.",
                ["5 V 출력, 500 mA 이상", "드롭아웃 ≤ 0.45 V (검증되지 않음)", "TO-252 열 성능 (8.4 V 에서 0.29–0.55 W)", "세라믹 출력 커패시터에 안정"],
                [unverified("BD50FC0FP (TO-252-2)", "핀 순서가 반대: 심볼을 바꿈"), unverified("IFX27001TFV50 (TO-252-3)"), unverified("LF50_TO252", "핀 순서가 반대")]),
    "U102": _ic("U102: TX_5V 레귤레이터 (V_TX → TX_5V, PA 전원)",
                "LM1117DT-5.0 (라이브러리 설명 '800mA'). 송신기에서 약 1 W 손실; 1.2 V 드롭아웃(검증되지 않음) 때문에 팩 하한 6.4 V 가 필요.",
                ["5 V 출력, 0.8 A 이상", "드롭아웃 ≤ 1.2 V", "TO-252 방열"], [unverified("LP38693DT-5.0", "441 mA 가 정격 안이면 팩 하한을 낮출 수 있음 - 사람의 설계 변경")]),
    "U103": _ic("U103: RX_3V3 저잡음 레귤레이터 (스켈치 비교기)", "LP5907MFX-3.3 (라이브러리 설명 '250-mA').", ["3.3 V, 저잡음", "SOT-23-5 핀 배열"], []),
    "U104": _ic("U104: TX_3V3 저잡음 레귤레이터 (송신 오디오, 논리)", "LP5907MFX-3.3 (라이브러리 설명 '250-mA').", ["3.3 V, 저잡음", "SOT-23-5 핀 배열"], []),
    "Q105": _ic("Q105: 주 전원 P-FET 스위치 (VBAT_F → V_SYS)",
                "AO3401A (라이브러리 설명 '-4.0A Id', TX 예산 441 mA 의 약 9배). SW101 은 게이트 전류(약 84 µA)만 흘려 PCM12 정격이 문제되지 않음(결정 3A).",
                ["P 채널, |I_D| ≥ 1 A", "|V_GS| 정격 ≥ 12 V (게이트가 팩 전압 8.4 V 를 봄, 검증되지 않음)", "R_DS(on) ≤ 100 mΩ at V_GS = −4.5 V"], [unverified("SI2301", "SOT-23, 핀 배열 확인")]),
    "Q101": _ic("Q101: RX 레일 P-FET 스위치", "AO3401A; 게이트가 PTT_ACTIVE (RX 에서 낮음).", ["P 채널, 문턱 |V_GS(th)| ≤ 1.3 V", "SOT-23 G/S/D 핀"], []),
    "Q102": _ic("Q102: TX 레일 P-FET 스위치", "AO3401A; Q104 가 게이트를 끌어내릴 때만 켜짐.", ["P 채널, |I_D| ≥ 1 A (TX 전류)"], []),
    "Q103": _ic("Q103: PTT 인버터 (PTT_N → PTT_ACTIVE)", "MMBT3904 범용 NPN (`model.npn`).", ["NPN, V_CEO ≥ 20 V (콜렉터가 V_SYS)"], [unverified("BC847", "SOT-23, 핀 배열 확인")]),
    "Q104": _ic("Q104: PTT 인버터 (PTT_ACTIVE → Q102 게이트)", "MMBT3904 범용 NPN.", ["NPN, V_CEO ≥ 20 V"], [unverified("BC847")]),
    "SW101": _ic("SW101: 전원 스위치 (Q105 게이트)", "SW_SPDT (PCM12): 극 B 가 게이트, A = GND 가 켜짐, C 는 개방.", ["게이트 전류만 흐름: 어떤 소형 슬라이드 스위치든 가능"], []),
    "J101": _ic("J101: 2S 팩 커넥터", "JST PH 2핀. 팩은 보호회로 내장, 외부 충전(기판에 충전기 없음).", ["1 = +, 2 = GND", "전류 ≥ 1 A"], []),
    "U201": _ic("U201: 송신 대기 지연 비교기", "LMV331 (오픈 컬렉터), TX_3V3 공급 - V_SYS 공급 금지(절대최대 약 5.5 V, 검증되지 않음).", ["단일 공급 3.3 V", "오픈 컬렉터 출력"], [unverified("TLV3011", "푸시풀 출력: 풀업 불필요, 핀 배열 확인")]),
    "U204": _ic("U204: 저전압 송신 금지 비교기", "LMV331; 팩 하한 6.4 V 를 하드웨어로 지킴(결정 3A).", ["단일 공급 3.3 V", "오픈 컬렉터 출력 (D202 로 DLY 를 잡음)"], []),
    "U202": _ic("U202: 송신 인터록 AND", "74LVC1G08, TX_3V3 공급. 입력은 5.5 V 허용(검증되지 않음)이므로 PTT_ACTIVE 는 절반으로 나눔.", ["2입력 AND, 3.3 V", "입력 과전압 허용"], []),
    "Q201": _ic("Q201: PA 전원 스위치 (TX_5V → PA_5V)", "AO3401A; 기본은 꺼짐(R216).", ["P 채널, |I_D| ≥ 1 A"], []),
    "Q202": _ic("Q202: PA 스위치 구동", "MMBT3904.", ["NPN"], []),
    "Q203": _ic("Q203: POWER_DOWN 인버터", "MMBT3904; PA_PD 높음 = PA 꺼짐(라이브러리 핀 이름 기준, 검증되지 않음).", ["NPN"], []),
    "SW201": _ic("SW201: PTT 버튼", "SW_Push (TL3342).", ["순간 접점, 정상 개방"], []),
    "U301": _ic("U301: 마이크 앰프 (AGC)", "MAX9814; GAIN / A/R / TH 스트랩은 검증되지 않은 선택값. 모델이 없어 설계 덱에서는 MICOUT 을 사인 자극으로 대체.",
                ["AGC 마이크 앰프, 2.7–5.5 V", "DFN-14 핀 배열"], [unverified("SSM2167", "압축기 내장, 핀·스트랩이 다름")]),
    "MK301": _ic("MK301: 일렉트릿 마이크",
                 "PUI Audio POM-2244P-C3310-2-R 스루홀 캡슐(Sensor_Audio:POM-2244P-C3310-2-R: THT 패드 1 / 2, 간격 1.9 mm, 드릴 0.65 mm). "
                 "심볼 Device:Microphone_Condenser 의 핀 1 '-' / 2 '+' 가 패드 1 / 2 에 붙고(- 는 GND, + 는 MIC_P), 풋프린트 실크의 '+' 표시는 패드 2 쪽이지만, "
                 "어느 단자가 케이스(음극)인지는 PUI Audio 데이터시트로 확인하지 않았습니다(검증되지 않음). 앞서 쓰던 SMT 캡슐 CUI CMC-4013 은 "
                 "중앙 패드 2 가 링 패드 1 안에 있어 패드 안 비아 없이는 F.Cu 로 빠져나올 수 없었습니다(2b / 2c 측정).",
                 ["2단자 일렉트릿, 바이어스 2.2 kΩ", "케이스(음극) 단자를 GND 에: 데이터시트로 극성 확인"], []),
    "D301": _ic("D301: 리미터 다이오드 (양의 반파)", "1N4148WS; 클리핑 레벨 ≈ 순방향 전압(선택값 0.6 V, 검증되지 않음).", ["소신호 실리콘 다이오드", "D302 와 짝"], []),
    "D302": _ic("D302: 리미터 다이오드 (음의 반파)", "1N4148WS.", ["D301 과 같은 부품"], []),
    "RV301": _ic("RV301: 편이 트림 (실험실 정렬)", "Bourns 3314J 트리머; 적분기 입력 레벨을 정함 - 편이는 변조 분석기로 맞춤.", ["10 kΩ 선형"], []),
    "U402": _ic("U402: 스피커 앰프", "LM386 (이득 핀 개방: 20, 검증되지 않음), V_RX 공급.", ["V_RX 4–12 V 동작", "SOIC-8 핀 배열"], [unverified("TDA7052", "BTL: 출력 커패시터 불필요, 핀이 다름")]),
    "U403": _ic("U403: 스켈치 비교기", "LMV331, RX_3V3 공급.", ["오픈 컬렉터 출력"], []),
    "RV401": _ic("RV401: 볼륨", "RK09K 패널 전위차계(마운팅 핀 GND).", ["10 kΩ 로그(audio) 또는 선형"], []),
    "RV402": _ic("RV402: 스켈치 문턱", "RK09K 패널 전위차계.", ["10 kΩ 선형"], []),
    "J401": _ic("J401: 스피커 커넥터", "JST PH 2핀.", ["8 Ω 스피커"], []),
    "J402": _ic("J402: 시험 헤더 (DISC_OUT, RSSI, GND)", "2.54 mm 3핀; 신호 발생기 또는 수신 백엔드에서 입력.", ["1 = DISC_OUT, 2 = RSSI, 3 = GND"], []),
    "J403": _ic("J403: 시험 헤더 (PM_DRIVE, PA_ON, GND)", "2.54 mm 3핀; 송신 여진기로 출력.", ["1 = PM_DRIVE, 2 = PA_ON, 3 = GND"], []),
    "J404": _ic("J404: 시험 헤더 (MUTE, PA_PD, GND)", "2.54 mm 3핀.", ["1 = MUTE, 2 = PA_PD, 3 = GND"], []),
    "U203": _ic("U203: 송신 시간 제한 카운터", "4060 (RC 발진 + 14단 카운터; Q14 = 2^13 주기). RC 상수는 검증되지 않음.", ["CD4060 / HEF4060 계열, 3 V 동작"], []),
}
for _ref, _what in (("U302", "고역통과 + 프리엠퍼시스"), ("U305", "리미터 버퍼"), ("U303", "스플래터 필터 1단 (Q 0.5412)"), ("U306", "스플래터 필터 2단 (Q 1.3066)"),
                    ("U307", "편이 트림 버퍼"), ("U304", "손실 적분기 (PM_DRIVE)"), ("U401", "고역통과 + 디엠퍼시스 (반전)"), ("U404", "저역통과 버퍼")):
    _PART_NOTES[_ref] = _ic(f"{_ref}: {_what}", "MCP6001-OT 단일 op-amp (critic2: 이중 MCP6002-xSN 은 다중 유닛 심볼이라 컴파일러가 거부). 설계 덱은 단극 매크로 `model.opamp`.",
                            ["GBW ≥ 1 MHz", "레일 투 레일 출력, 3.3–5 V 단일 공급", "SOT-23-5 핀 배열 (1 = 출력)"],
                            [unverified("TLV9001", "SOT-23-5, 핀 배열 확인"), unverified("MCP6L01")])


KR447_AUDIO_PTT = Kr447AudioPttTemplate()

__all__ = [
    "BANDWIDTH_RANGE",
    "BUILD",
    "DEVIATION_RANGE",
    "KR447_AUDIO_PTT",
    "REGIONS",
    "TEMPLATE_ID",
    "V_IN_MAX",
    "V_IN_MIN",
    "STAGE1_BLOCK_IDS",
    "BenchHeadersBlock",
    "Kr447AudioPttTemplate",
    "stage1_blocks",
    "stage1_part_notes",
    "stage1_refs",
]
