"""The ``uhf_fm_breadboard`` template: a discrete UHF FM walkie-talkie for a perfboard / breadboard experiment in the amateur 430-440 MHz band.

Selected by the categorical requirement ``radio_build = uhf_breadboard``,
needing ``carrier_frequency`` (430-440 MHz: one fixed channel, the crystal is
f_c / 12), ``modulation`` (FM) and ``input_voltage`` (7.5-12 V: a 9 V
battery), optionally ``audio_bandwidth`` (2-4 kHz). Every other confirmed
design requirement refuses the board with the family's closed-world sentence
and the ``--answer leave_out=<key>`` that leaves it out.

The user's decisions (2026-11-03): no programmable parts - transistors and
op-amps wherever possible; an amateur licence, so the board transmits only
under it, in 430-440 MHz [UNVERIFIED: the licence's band plan and power];
the circuit, a perfboard layout and this pipeline's KiCad board. The design
studies (ngspice-42, generic cards, :mod:`ai_eda.design.rf.blocks.breadboard`)
chose a crystal VXO x12 over a 433.92 MHz SAW oscillator: with a 5 GHz
transistor both SAW topologies tried either free-ran at 505 MHz through the
SAW's C0 or did not start, while a fundamental crystal at 36 MHz has about
three times the start-up margin; and a self-quenched super-regenerative
detector (about 190 kHz quench) as the receiver, the simplest one that
works with a wide frequency tolerance, slope-detecting narrow-band FM.

The board is six blocks of :mod:`ai_eda.design.rf.blocks.breadboard`
re-based by hundreds (power 1xx, tx_audio 2xx, tx_rf 3xx, rx 4xx, rx_audio
5xx, trx 6xx) in floorplan regions (``floor.*`` choices): the regions are
also the perfboard's sections. Two layers by default (a perfboard has one
copper side; the KiCad board puts ground on the back), four allowed.

What the checks prove is principle under confirmed model values: the audio
chains' responses, the references and the output stage's DC point and gain
in the design deck; the multiplier tanks, the band-pass, the PA match, the
low-pass and the T/R network as RF fixtures. The oscillator, the multipliers'
drive, the PA's power, the spurious emissions, the super-regenerative
detector and the speaker are lab items. Transmitting needs the amateur
licence's conditions to be met first (spurious emissions measured into a
dummy load).
"""

from __future__ import annotations

from ai_eda.ir import Block, CircuitDomain, CircuitIR, Constraint, ConstraintKind, MissingInformation, SimulationSetup, Topology, Traced
from ai_eda.ir.rf import RFBlock, RFDesign, RFRegion
from ai_eda.tools.kicad.library import KicadLibrary

from ai_eda.design.base import (
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
from ai_eda.design.rf.blocks.base import BlockBuilder, BlockContext, BlockPrefix, BlockResult, exclude_floating, merge_results
from ai_eda.design.rf.blocks.breadboard import (
    F_C_RANGE_HZ,
    V_IN_RANGE,
    BbPowerBlock,
    BbRxAudioBlock,
    BbRxBlock,
    BbTrxBlock,
    BbTxAudioBlock,
    BbTxRfBlock,
)
from ai_eda.design.rf.family import BUILDS, refusal_question, unserved_message
from ai_eda.design.rf.models import MODEL_VERDICT
from ai_eda.design.rf.parts import part_line
from ai_eda.design.templates import _changes, _missing_inputs, _net_line, _refused

TEMPLATE_ID = "uhf_fm_breadboard"
RADIO_BUILD = "uhf_breadboard"
BUILD = BUILDS[RADIO_BUILD]
#: the audio bandwidth the Sallen-Key low-passes are designed for (Hz)
BANDWIDTH_RANGE = (2000.0, 4000.0)
LAYER_REASON = "a perfboard has one copper side: 2 layers by default (ground on the back of the KiCad board); 4 allowed"
#: block id -> (x, y, w, h) mm of its floorplan region (origin top-left, Y down): also the perfboard's sections, the RF chain along the bottom
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "power": (0.0, 0.0, 50.0, 45.0),
    "tx_audio": (50.0, 0.0, 70.0, 45.0),
    "rx_audio": (120.0, 0.0, 80.0, 60.0),
    "tx_rf": (0.0, 45.0, 120.0, 55.0),
    "trx": (120.0, 60.0, 35.0, 40.0),
    "rx": (155.0, 60.0, 45.0, 40.0),
}
_REGION_WHAT = {
    "power": "battery header, switches and the two L78L05 at the top left",
    "tx_audio": "microphone amplifier, low-pass and deviation trimmer at the top centre",
    "rx_audio": "RX audio, volume and the BD139 / BD140 output stage at the top right",
    "tx_rf": "VXO, multipliers, band-pass, PA and low-pass along the bottom left (the RF chain left to right)",
    "trx": "T/R switch and the SMA at the bottom centre-right",
    "rx": "RF amplifier and super-regenerative detector at the bottom right",
}
#: the blocks in build order (the TX RF block writes the plan values the T/R and RX blocks read) and their numbering
BLOCKS = (
    (BbPowerBlock(), BlockPrefix(ref_base=100)),
    (BbTxAudioBlock(), BlockPrefix(ref_base=200)),
    (BbTxRfBlock(), BlockPrefix(ref_base=300)),
    (BbTrxBlock(), BlockPrefix(ref_base=600)),
    (BbRxBlock(), BlockPrefix(ref_base=400)),
    (BbRxAudioBlock(), BlockPrefix(ref_base=500)),
)


def _board_values(ctx: BlockContext) -> BlockResult:
    """The rows that belong to the board: the floorplan regions."""
    b = BlockBuilder(ctx, "board", "board values")
    for bid, box in REGIONS.items():
        b.choice(f"floor.{bid}", list(box), "mm", f"floorplan region of the {bid} block (x, y, w, h mm from the top-left corner): {_REGION_WHAT[bid]}")
    return b.done()


class UhfFmBreadboardTemplate(Template):
    """A discrete UHF FM walkie-talkie for a perfboard (module docstring)."""

    id = TEMPLATE_ID
    title = "UHF FM walkie-talkie for a perfboard (amateur 430-440 MHz, discrete)"
    triggers = ()
    needs = BUILD.needs
    serves = BUILD.serves
    layer_policy = LayerPolicy(allowed=(2, 4), default=2, reason=LAYER_REASON)
    plane_nets = ("GND", None)

    def triggered_by(self, ir: CircuitIR, inputs: dict[str, DesignInput]) -> bool:
        return read_radio_build(ir)[0] == RADIO_BUILD

    def refusals(self, ir: CircuitIR, inputs: dict[str, DesignInput], unusable: dict[str, str]) -> list[MissingInformation]:
        out: list[MissingInformation] = []
        modulation, _why = read_modulation(ir)
        if modulation is not None and modulation != "fm":
            why = f"modulation {modulation}: radio_build={RADIO_BUILD} builds a varactor-FM transmitter and a slope-detecting receiver only"
            out.append(MissingInformation(key="modulation", required=False, rationale=why,
                                          question=with_remedy(f"{why}; no template design was proposed. State modulation=FM or choose another design",
                                                               unusable_remedy(ir, MODULATION_ALIASES)) + "."))
        unserved = unserved_requirements(ir, self)
        keys = [r.key for r in unserved]
        for r in unserved:
            canon = canonical_key(r.key) or r.key
            why = f"{r.id} ({requirement_text(r)}): {unserved_message(canon, RADIO_BUILD)}"
            out.append(refusal_question(why, r.key, RADIO_BUILD, keys))
        return out

    @staticmethod
    def _out_of_range(inputs: dict[str, DesignInput]) -> str | None:
        for key, (lo, hi), why in (
            ("carrier_frequency", F_C_RANGE_HZ, "the amateur 70 cm band the board is designed for (one channel; the crystal is f_c / 12) [UNVERIFIED: the licence's band plan]"),
            ("input_voltage", V_IN_RANGE, "a 9 V battery or a 12 V supply: the L78L05 behind the 1N5817 needs about 7.5 V, the BFR92A about 15 V at most"),
            ("audio_bandwidth", BANDWIDTH_RANGE, "the Sallen-Key low-passes' design range (a voice channel)"),
        ):
            inp = inputs.get(key)
            if inp is not None and not lo <= float(inp.traced.value) <= hi:
                return f"{key} {float(inp.traced.value):.12g} {inp.traced.unit} ({inp.requirement.id}) is outside {lo:.12g}..{hi:.12g} {inp.traced.unit}: {why}"
        return None

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
            _missing_inputs(plan, f"The {self.title} template", missing, unusable, examples={"carrier_frequency": "433.5 MHz", "input_voltage": "9 V"}, ir=ir)
        modulation, mod_why = read_modulation(ir)
        if modulation is None:
            plan.questions.append(MissingInformation(key="modulation", required=mod_why is None, rationale="template input",
                                                     question=with_remedy(mod_why, unusable_remedy(ir, MODULATION_ALIASES)) or f"The {self.title} template needs modulation: answer modulation=fm"))
            if not missing:
                return _refused(plan, f"modulation missing{': ' + mod_why if mod_why else ''}")
        if plan.questions or plan.notes:
            return plan
        if modulation != BUILD.modulation:
            return _refused(plan, f"modulation {modulation!r}: this board transmits and receives FM only")
        plan.inputs = stated
        ctx = BlockContext(ir=ir, library=library, template_id=t, confirmed=confirmed, inputs=inputs)
        results: list[BlockResult] = []
        try:
            for blk, prefix in BLOCKS:
                r = blk.build(ctx, prefix)
                results.append(r)
                ctx.shared.update({k: v for k, v in r.params.items() if k not in ctx.shared})
            merged = merge_results([_board_values(ctx), *results], t)
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
                          rails=list(merged.rails), model_values=sorted(merged.model_keys), profile_keys=[])
        except ValueError as e:
            return _refused(plan, f"the RF design refuses the composition: {e}")
        plan.choices = list(merged.choices)
        plan.computed = list(merged.computed)
        plan.parts = [part_line(merged.placed[c.ref]) for c in components if c.ref in merged.placed]
        plan.nets = [_net_line(n) for n in merged.nets]
        sim = SimulationSetup(stimuli=list(merged.stimuli), analyses=list(merged.analyses), expectations=list(merged.expectations))
        plan.simulation = self._simulation_lines(sim, components, report)
        plan.notes.extend(merged.notes)
        topology = Topology(
            name=self.title, domains=[CircuitDomain.RF, CircuitDomain.ANALOG, CircuitDomain.POWER],
            rationale=("discrete UHF FM walkie-talkie for a perfboard: 9 V -> TX / RX switch -> two L78L05; TX: electret -> TL071 gain with a diode soft "
                       "limiter -> TL071 Sallen-Key low-pass -> deviation trimmer -> varactor in series with a fundamental crystal at f_c / 12 in a 2N3904 "
                       "Colpitts VXO (direct FM) -> BF199 x3 -> BFR92A x2 -> BFR92A x2 -> 3-resonator top-C band-pass -> BFR92A class-A PA -> L-match -> "
                       "5th-order Chebyshev low-pass -> 1N4148 T/R switch -> SMA; RX: BFR92A common-base RF amplifier -> self-quenched BFR92A "
                       "super-regenerative detector -> TL071 gain -> TL071 low-pass -> volume -> TL071 + BD139 / BD140 -> speaker. No programmable part"),
            provenance=ctx.provenance(f"selected by radio_build = {RADIO_BUILD}"),
            blocks=[Block(id=r.block_id, function=r.title, domain=CircuitDomain.RF if r.block_id in ("tx_rf", "trx", "rx") else
                          CircuitDomain.POWER if r.block_id == "power" else CircuitDomain.ANALOG,
                          component_refs=[c.ref for c in r.components], provenance=ctx.provenance(f"block {r.block_id}")) for r in results],
        )
        plan.changes = _changes(t, self.title, topology, components, merged.nets, dict(merged.params), sim, self._constraints(ctx))
        plan.changes.append(DesignChange(description="RF design: blocks, fixture networks, frequency plan, lab items", target="rf", operation="set", payload=rf,
                                         rationale=f"template {t}: {len(rf.networks)} fixture network(s), {len(rf.frequency_plan)} plan row(s), {len(rf.lab_items)} lab item(s)"))
        return plan

    # ------------------------------------------------------------------ build steps

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
            if e.bound:
                rule = f"{e.bound} {e.nominal.value:.6g}"
            else:
                tol = e.tol_abs if e.tol_abs is not None else e.tol_rel
                rule = f"{e.nominal.value:.6g} +/- {tol.value:.3g}" + ("" if e.tol_abs is not None else " (relative)")  # type: ignore[union-attr]
            at = f" at {e.at.value:.6g}" if e.at is not None else ""
            ref = f" re {e.reference_vector}" if e.reference_vector else ""
            lines.append(f"expectation {e.id}: {e.vector}{ref} {e.reduce.value}{at} {rule} {e.nominal.unit or ''} ({e.provenance.note})")
        excluded = [c.ref for c in components if c.spice is not None and c.spice.exclude]
        lines.append(f"excluded from the netlist ({len(excluded)}): {', '.join(excluded)} (reasons in each part's SPICE binding; "
                     f"{len(report.excluded)} of them dead branches found by exclude_floating)")
        lines += [f"reported, kept: {f.reason}" for f in report.unresolved]
        lines.append(f"every PASS here is a {MODEL_VERDICT}: no RF transistor, crystal or regulator is simulated")
        return lines

    @staticmethod
    def _constraints(ctx: BlockContext) -> list[Constraint]:
        s = ctx.provenance
        return [
            Constraint(id="c.bb.amateur_licence", kind=ConstraintKind.REGULATORY, target="*", provenance=s("the user's decision: an amateur licence"),
                       description=("the board transmits only under the user's amateur radio licence, in 430-440 MHz, within its power and emission class "
                                    "[UNVERIFIED: the licence's band plan, the KR amateur station rules]; a self-built transmitter is never used on the "
                                    "licence-exempt 447 MHz channels (they need KC-certified equipment)")),
            Constraint(id="c.bb.dummy_load_first", kind=ConstraintKind.REGULATORY, target="J601", provenance=s("spurious emissions"),
                       description=("the transmitter is aligned and its spurious emissions (f_c -/+ f_X, the harmonics) are measured into a 50 ohm dummy "
                                    "load before any antenna is connected (lab item tx_spurious)")),
            Constraint(id="c.bb.rf_leads", kind=ConstraintKind.MECHANICAL, target="tx_rf", provenance=s("UHF construction"),
                       description=("at 432 MHz every lead counts (1 mm of wire is about 1 nH): the RF stages are built with the shortest leads, ground "
                                    "returns to a ground plane or copper-clad strip, and never on a solderless breadboard (only the audio and power "
                                    "sections are breadboard-safe)")),
        ]

    # ------------------------------------------------------------------ report hooks (views)

    def theory(self, ir: CircuitIR) -> list[TheorySection]:
        p = lambda k: parameter_value(ir, k)  # noqa: E731
        f_c, f_x = p("rf.f_c"), p("tx.f_xtal")
        return [
            TheorySection("구성", (
                "이 무전기는 프로그래머블 소자 없이 트랜지스터와 연산증폭기만으로 만든 UHF FM 송수신기입니다. 송신은 수정 VXO(전압 제어 수정 발진기)에 "
                "가변용량 다이오드를 직렬로 달아 직접 FM을 만들고, 체배기 세 단(×3, ×2, ×2)으로 12배 해 반송파를 얻습니다. 수신은 공통 베이스 RF "
                "증폭기 뒤의 초재생(super-regenerative) 검파기 하나로, 반송파에서 조금 비켜 맞추면 협대역 FM을 기울기 검파합니다. 송수신 전환은 "
                "TX/RX 슬라이드 스위치가 전원을 바꾸고, 1N4148 두 개와 집중정수 λ/4가 안테나를 전환합니다.")),
            TheorySection("주파수 계획", (
                f"반송파 f_c = {quantity(f_c, 'Hz', 7)}, 수정 주파수 f_X = f_c / 12 = {quantity(f_x, 'Hz', 8)}. 체배기 출력은 3 f_X = "
                f"{quantity(p('tx.f1'), 'Hz', 7)}, 6 f_X = {quantity(p('tx.f2'), 'Hz', 7)}, 12 f_X = {quantity(p('tx.f3'), 'Hz', 7)}입니다. "
                f"가장 가까운 불요파는 f_c ± f_X = {quantity(p('tx_bpf.f_m'), 'Hz', 7)} / {quantity(p('tx_bpf.f_p'), 'Hz', 7)}이고, 3공진기 대역통과 "
                f"필터의 계산값은 각각 {number(p('tx_bpf.rel_m'), 4)} dB / {number(p('tx_bpf.rel_p'), 4)} dB(코일 Q {number(p('model.bb.coil_q'), 3)} 가정)입니다.")),
            TheorySection("FM 편이", (
                "수정의 부하 용량 C_L이 바뀌면 직렬 공진 주파수가 Δf/f ≈ C_m / (2 (C_0 + C_L)) 만큼 끌려갑니다. 가변용량 다이오드의 역바이어스에 "
                "오디오를 실으면 f_X가 수백 Hz 움직이고, 12배 뒤에는 수 kHz의 편이가 됩니다. 편이 크기는 편이 트리머(RV201)로 맞춥니다. "
                f"가변용량 다이오드 바이어스 설계값은 {quantity(p('tx.v_bias'), 'V', 4)}입니다(실제 편이는 실험 항목).")),
            TheorySection("초재생 검파", (
                "검파기는 스스로 발진을 키우다 이미터의 RC로 꺼지는 동작(자기 소멸)을 수백 kHz로 반복합니다. 입력 신호가 클수록 발진이 일찍 "
                f"시작해 소멸 노드의 평균 전압이 바뀌고, 그것이 오디오가 됩니다. 소멸 주파수 예상값은 {quantity(p('rx.sr.f_quench'), 'Hz', 3)}"
                "(일반 트랜지스터 모델로 ngspice-42 과도해석)이며, 오디오 저역통과 필터가 이를 걸러 냅니다. 재생 트리머(RV401)는 '쉬' 하는 잡음이 "
                "막 들리기 시작하는 지점에 맞춥니다.")),
            TheorySection("만능기판과 브레드보드", (
                "오디오 단과 전원부는 브레드보드에서 먼저 시험해도 됩니다. VXO 이후의 RF 단(36 MHz부터 432 MHz)은 브레드보드에서 동작하지 않으므로, "
                "만능기판(가능하면 동판을 접지면으로 깐 기판)에 리드를 최대한 짧게 납땜합니다. BFR92A는 SOT-23이라 SOT-23→DIP 변환 기판에 올려 "
                "꽂습니다. 대역통과 필터의 0.2 pF 결합은 절연 전선 두 가닥을 꼬아 만든 '기믹' 커패시터로 조정합니다.")),
        ]

    def part_notes(self, ir: CircuitIR) -> dict[str, PartNote]:
        f_x = parameter_value(ir, "tx.f_xtal")
        return {
            "Y301": PartNote(role="VXO 수정(f_c / 12)", why=f"기본파 모드 HC-49/U, {quantity(f_x, 'Hz', 8)}. 36.000 MHz 기성품이면 432.000 MHz가 됩니다.",
                             criteria=["기본파(fundamental) 모드", "주문 시 부하 용량과 끌림 범위 명시"], substitutes=[unverified("주문 제작 수정", "f_c / 12")]),
            "D301": PartNote(role="VXO 가변용량 다이오드", why="수정과 직렬로 부하 용량을 바꿔 직접 FM을 만듭니다.",
                             criteria=["2-4 V 역바이어스에서 20-40 pF 부근"], substitutes=[unverified("BB910"), unverified("1SV101"), unverified("BB109")]),
            "Q302": PartNote(role="×3 체배기", why="BF199(f_T 약 550 MHz, TO-92)는 108 MHz 출력에 충분합니다.", criteria=["f_T 500 MHz 이상"],
                             substitutes=[unverified("BF240"), unverified("2SC9018")]),
            "Q303": PartNote(role="×2 체배기", why="BFR92A(f_T 약 5 GHz)", criteria=["f_T 2 GHz 이상"], substitutes=[unverified("BFR93A"), unverified("2SC3356")]),
            "Q304": PartNote(role="×2 체배기(432 MHz 출력)", why="BFR92A", criteria=["f_T 2 GHz 이상"], substitutes=[unverified("BFR93A"), unverified("2SC3356")]),
            "Q305": PartNote(role="A급 출력단(약 30 mW 설계)", why="BFR92A를 약 16 mA로 바이어스", criteria=["P_tot 300 mW 이상", "I_C 25 mA 이상"],
                             substitutes=[unverified("BFR93A"), unverified("BFR96")]),
            "Q402": PartNote(role="초재생 검파기", why="BFR92A 자기 소멸 콜피츠", criteria=["f_T 2 GHz 이상"], substitutes=[unverified("BFR93A"), unverified("2SC3356")]),
            "U501": PartNote(role="RX 오디오 증폭", why="TL071 단일 연산증폭기(DIP-8)", criteria=["단일 9 V 전원에서 중간 기준 전압 동작"],
                             substitutes=[unverified("TL081"), unverified("LF351")]),
        }


UHF_FM_BREADBOARD = UhfFmBreadboardTemplate()

__all__ = ["RADIO_BUILD", "REGIONS", "TEMPLATE_ID", "UHF_FM_BREADBOARD", "UhfFmBreadboardTemplate"]
