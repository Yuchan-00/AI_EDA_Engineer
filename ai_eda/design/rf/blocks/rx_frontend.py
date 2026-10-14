"""The receive front end of the KR 447 MHz FM radio (kr447 design §2.3, part P11): BPF -> BFR92 LNA -> BPF -> ADEX-10 -> diplexer -> IF1 post-amp.

Invariant: every part comes from the kr447 parts table by library name
(:mod:`ai_eda.design.rf.parts`), every number is a registered calculator's
output over the blocks' confirmed choices, the ``model.*`` values
(UNVERIFIED, listed in ``ir.rf.model_values``) and the confirmed
requirements, and every passive network the excluded ICs and transistors see
is an RF fixture (``ir.rf.networks``) whose rows the fixture runner judges on
ngspice. Nothing here says that the LNA, the mixer or the receiver work: the
BFR92 has no S-parameters (its ports are the ``model.*.port_r`` resistances,
its match is not designed), the ADEX-10 has no model (excluded from every
netlist), and a fixture PASS is "a network verdict under confirmed model
values (not a measured part)" at schematic level - no track, via or
ground-return inductance.

Two blocks, because a floorplan block is one region and a shield can holds
every part of its block (``placement.rf_floorplan``):

* :class:`RxFrontendBlock` (``rx_frontend``, local references 1..49, re-based
  to 6xx by the template) - under the can ``SH1`` (a
  ``Device:RFShield_OnePiece``; which Laird can is a constructor argument):
  ``RX_RF`` -> the 2-pole top-C band-pass ``fe_bpf2`` (``C1`` .. ``C5``,
  ``L1`` / ``L2``) -> ``LNA_IN`` -> the BFR92 LNA ``Q1`` (divider bias
  ``R1`` / ``R2``, emitter ``R3`` // ``C6``, collector choke ``L3`` fed from
  ``RX_5V`` through the 0 ohm link ``R4`` with its decoupling ``C7``) ->
  ``LNA_OUT`` -> the 3-pole top-C band-pass ``fe_bpf3`` (``C8`` .. ``C14``,
  ``L4`` .. ``L6``) -> ``MIX_RF``;
* :class:`RxMixerBlock` (``rx_mixer``, local references 50..99, re-based to
  6xx): the ADEX-10 ``U50`` (RF ``MIX_RF``, LO ``LO1_MIX``, IF ``MIX_IF``,
  the three stacked GND pins in one net) -> the diplexer ``diplexer`` (the
  series ``L50`` - ``C50`` resonant at IF1 to the post-amp; ``C51`` + the
  absorptive ``R50`` to ground for the LO / RF / sum products) -> the BFR92
  IF1 post-amplifier ``Q50`` (bias as the LNA's, choke ``L51``, link
  ``R54``, output DC block ``C54``) -> ``IF1``. The diplexer's fixture
  loads it with the post-amp's base as the board does: the port model
  ``model.ifamp.port_r`` (the transistor's own input) beside the base
  divider ``R51`` / ``R52`` (members; the divider costs the IF1 row about
  0.16 dB: -0.462 dB against -0.299 dB without it, bound -1 dB). The
  post-amp's output into ``IF1`` is not matched: the IF1 port declares the
  system impedance, which only the lab item ``ifamp_output`` can confirm.

Interface nets (never prefixed): ``RX_RF`` (the 50 ohm RF input: the stage
board's U.FL, the transceiver's T/R switch), ``MIX_RF`` (between the two
blocks), ``LO1_MIX`` (from the LO chain's pad), ``IF1`` (to the IF
back-end's input, the name part P10 uses), ``RX_5V`` and ``GND``.

The filters are Butterworth top-C coupled-resonator networks designed by
``calc.rf.resonator.top_c.*`` (Dishal / Zverev: capacitive end taps, top
coupling capacitors, shunt capacitors) between their *loaded* ports, and
every part the board hangs on their port nets is a member of their fixture
(the kr447 fixture-membership pass, part A1; ``tests/test_rf_fixture_members.py``
audits it on every build):

* ``fe_bpf2`` runs from the system impedance at ``RX_RF`` into the LNA's base:
  the port model ``model.lna.port_r`` (the transistor's own input) in
  parallel with the base divider ``R1`` / ``R2`` (``fe.lna.r_div``, members,
  ``R1`` returned to ``RX_5V`` - the fixture's rail port, an ac short). The
  network is designed for ``fe_bpf2.r_load_eff`` = port // divider, and its
  s21 row reads the power reaching the port model: the network's own S21
  into r_load_eff (``fe_bpf2.s21_net``) plus the share
  ``fe_bpf2.load_share_db`` = 10 log10(R_div / (R_port + R_div))
  (``calc.divider.ratio``, ``calc.rf.power_ratio_to_db``; -0.1615 dB at the
  default choices), ``calc.rf.db_sum``;
* ``fe_bpf3`` runs from the LNA's collector port - ``model.lna.port_r`` in
  parallel with the collector feed choke ``L3`` (its Q ``model.l_q.uhf``,
  returned to ``RX_5V`` through the 0 ohm link ``R4``; ``R4`` and the feed
  decoupling ``C7`` are members too) whose reactance the input tap absorbs
  (``calc.rf.resonator.top_c.port_r`` / ``.port_x`` / ``.c_tap_reactive``,
  the rows ``.ported_s21_db`` / ``.ported_rel_s21_db``) - into
  ``model.adex10.port_r`` at the mixer.

Every inductor carries the series loss w0 L / Q_u of ``model.l_q.uhf``.
Every fixture row's nominal is the exact network response (tol_abs
``fe.net_tol``): at these offsets a top-C network is asymmetric, and the
symmetric narrowband formulas would FAIL a correct network (kr447 design
§2.7). At the default choices (4.7 nH, 40 / 20 MHz, Q_u 40, 50 ohm) the
calculators give ``fe_bpf2`` -3.529 dB at f_c and 13.565 dB image
rejection, ``fe_bpf3`` -9.322 dB, 33.597 dB image and 15.426 dB LO1
rejection (ngspice-42 reads the same to 1e-4 dB; before the divider and the
choke were members the fixtures gave -3.367 / -9.303 dB for a network the
board does not have). The design's one-sided bounds for these rows
(at_least -4 / -10 dB, image at_most -12.5 / -32.5 dB) are what the same
model gives less about 1 dB; the exact nominal +/- 1 dB judges the same
network and also catches a netlist that realises another one.

Every filter part is a fixture member only: the design deck leaves it out
(its series taps would leave DC-floating nodes between them) and the network
binds it at its calculator value. The design deck is the bias of the two
BFR92 stages under ``model.npn`` (``op_bias``): I_C through the 0 ohm link
(simulated as a 0 V source, an ideal ammeter - exact for a 0 ohm link) at
the nominal of ``calc.rf.bjt_bias.ic`` (V_B from ``calc.divider.v_out``,
V_BE a choice), tol_rel ``fe.bias_tol`` - a principle check of the bias
network under a generic transistor card, never the LNA's gain or NF.

The noise-figure budget (:class:`RxMixerBlock`): Friis from the back -
``model.ifb.nf`` (the IF back-end), the post-amp (``model.ifamp.nf`` /
``.gain``), the mixer's conversion loss ``model.adex10.cl``, ``fe_bpf3``'s
Cohn loss (``calc.rf.bpf.dissipation_loss``, a passive loss adds in dB:
``calc.rf.db_sum``), the LNA (``model.lna.nf`` / ``.gain``,
``calc.rf.friis_nf``), ``fe_bpf2``'s Cohn loss - and the sensitivity
kTB + NF + SNR (``calc.rf.sensitivity``) at ``rf.if_bw`` and ``fe.snr``.
It is an estimate under assumed device numbers (kr447 decision 2A: about
-113 dBm for the transceiver, whose LPF and T/R switch this bench board has
not), recorded as parameters and as the lab item ``sensitivity``, never a
verdict; a stated ``rx_sensitivity`` is named there, never refused and never
PASS.

Plan values shared with other blocks (``rf.f_c``, ``rf.if1``,
``rf.lo1_side``, ``rf.lo1``, ``rf.z0``, ``rf.if_bw``) are taken from
:attr:`BlockContext.shared` when a composing template wrote them, else
written here; the rail level ``RX_5V`` is the power block's ``power.rx_5v``
(or a stand-in's), read from :attr:`BlockContext.shared` - a block without it
refuses. The profile's ``kr447.channel_raster`` is read from the shared
profile choices.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from ai_eda.ir import AnalysisSpec, Expectation, NetKind, Reduce, SpiceBinding, SpiceDevice, Traced
from ai_eda.ir.rf import LabItem, PlanLine, RFExpectation, RFNetwork, RFPort, RFProbe
from ai_eda.tools.calc import radio
from ai_eda.tools.calc.basic import parallel_resistance, voltage_divider_output, voltage_divider_ratio
from ai_eda.tools.calc.part_value import format_part_value
from ai_eda.tools.calc.rf import friis_nf, lc_c_for_resonance, power_ratio_to_db, sensitivity
from ai_eda.tools.spice import SpiceAnalysis

from ai_eda.design.inputs import canonical_key
from ai_eda.design.library_parts import TemplateRefusal
from ai_eda.design.rf.blocks.base import GROUND_NET, Block, BlockBuilder, BlockContext, BlockResult
from ai_eda.design.rf.models import ModelValue, card_binding, inductor_q_key, npn_card
from ai_eda.design.rf.parts import PlacedPart

#: block ids (``ir.rf.blocks`` / the networks' ``block``)
FRONTEND_ID = "rx_frontend"
MIXER_ID = "rx_mixer"
#: the interface nets each block keeps (never prefixed)
FRONTEND_INTERFACE: tuple[str, ...] = ("RX_RF", "MIX_RF", "RX_5V")
MIXER_INTERFACE: tuple[str, ...] = ("MIX_RF", "LO1_MIX", "IF1", "RX_5V")
#: the fixture networks each block declares
FRONTEND_NETWORKS: tuple[str, ...] = ("fe_bpf2", "fe_bpf3")
MIXER_NETWORKS: tuple[str, ...] = ("diplexer",)
#: the rail nets and the power block's parameter that holds each level (read from BlockContext.shared)
RAIL_KEYS: dict[str, str] = {"RX_5V": "power.rx_5v", "RX_3V3": "power.rx_3v3"}
#: the profile value the blocks read from BlockContext.shared
RASTER_KEY = "kr447.channel_raster"
#: the LO-spur response ids the LO chain computes (``lo.<id>``): probes of the front-end filters when shared
LO_SPUR_RESPONSES: tuple[str, ...] = ("lo_spur_p1m", "lo_spur_p2m", "lo_spur_m1p", "lo_spur_p1p")
#: the rail every RF stage of these blocks is fed from
RAIL_NET = "RX_5V"
#: the value text of each shield-can row (the Laird part number)
SHIELD_VALUE: dict[str, str] = {"shield_102": "BMI-S-102", "shield_103": "BMI-S-103", "shield_105": "BMI-S-105"}
#: the design-deck analysis every bias expectation of the RF transistors reads
BIAS_ANALYSIS = "op_bias"
BIAS_NOTE = "operating point: the RF transistors' bias (I_C through the 0 ohm links)"

#: the model values of this module (port resistances of excluded parts, noise / gain numbers of the budget)
LNA_PORT = ModelValue("model.lna.port_r", 50.0, "ohm",
                      ("the BFR92 LNA transistor's own input resistance at its base and output resistance at its collector, as the front-end filters see "
                       "them (its match is not designed: no S-parameters exist here); the base divider and the collector feed choke are not inside it - "
                       "they are fixture members beside the port"),
                      "NXP BFR92AW S-parameters and a VNA measurement of the built LNA")
ADEX_PORT = ModelValue("model.adex10.port_r", 50.0, "ohm", "RF / LO / IF port resistance of the ADEX-10 mixer (excluded from every netlist)",
                       "Mini-Circuits ADEX-10 datasheet")
IFAMP_PORT = ModelValue("model.ifamp.port_r", 50.0, "ohm",
                        ("the BFR92 IF1 post-amplifier transistor's own input resistance at its base (the diplexer's load; its match is not designed: no "
                         "S-parameters exist here); the base divider is not inside it - it is a fixture member beside the port"),
                        "NXP BFR92AW S-parameters and a VNA measurement of the built post-amplifier")
LNA_NF = ModelValue("model.lna.nf", 2.0, "dB", "noise figure of the BFR92 LNA (sensitivity budget only)", "NXP BFR92AW datasheet and an NF-meter measurement")
LNA_GAIN = ModelValue("model.lna.gain", 15.0, "dB", "gain of the BFR92 LNA (sensitivity budget only)", "NXP BFR92AW S-parameters and a VNA measurement")
ADEX_CL = ModelValue("model.adex10.cl", 7.0, "dB", "conversion loss of the ADEX-10 at +7 dBm LO (its noise figure in the budget)", "Mini-Circuits ADEX-10 datasheet")
IFAMP_NF = ModelValue("model.ifamp.nf", 3.0, "dB", "noise figure of the BFR92 IF1 post-amplifier (sensitivity budget only)", "NXP BFR92AW datasheet and an NF-meter measurement")
IFAMP_GAIN = ModelValue("model.ifamp.gain", 15.0, "dB", "gain of the BFR92 IF1 post-amplifier (sensitivity budget only)", "NXP BFR92AW S-parameters and a VNA measurement")
IFB_NF = ModelValue("model.ifb.nf", 5.0, "dB", "noise figure of the IF back-end at its IF1 input (the SA605 receiver of radio_build rx_backend; sensitivity budget only)",
                    "NXP SA605 datasheet and a SINAD measurement of the rx_backend board")

#: why a fixture-only part is not in the design deck
FIXTURE_ONLY = ("an RF network part: simulated only in its block's RF fixture ({network}) at its calculator value - in the design deck its series "
                "taps would leave DC-floating nodes, and the deck judges only the transistors' bias")


# --------------------------------------------------------------------------- shared helpers (lo_chain imports these)


def part_value(x: float) -> str:
    """A part value as KiCad writes it (``4.7n``, ``3.3k``)."""
    return format_part_value(float(x))


def plan_value(b: BlockBuilder, key: str, make: Callable[[], Traced]) -> Traced:
    """A plan-level value: the composing template's (``ctx.shared``) when it wrote one, else ``make()`` (which writes it in this block)."""
    have = b.ctx.shared.get(key)
    return have if have is not None else make()


def copy_input(b: BlockBuilder, key: str, traced: Traced) -> Traced:
    """A confirmed requirement copied into a parameter (``design.inputs_vs_requirements`` re-reads it), unless a composing block already did."""
    return plan_value(b, key, lambda: b._param(key, traced))


def rail_level(b: BlockBuilder, net: str) -> tuple[str, Traced]:
    """``(key, level)`` of a rail the power block (or its bench stand-in) writes; refuses when the composition did not share it."""
    key = RAIL_KEYS[net]
    t = b.ctx.shared.get(key)
    if t is None:
        raise TemplateRefusal(f"block {b.result.block_id}: the level of {net} is the power block's parameter {key!r} (or a bench stand-in's), which the composing "
                              "template did not share (BlockContext.shared): compose the supply block first")
    return key, t


def requirement_ids(ctx: BlockContext, *keys: str) -> list[str]:
    """The ids of the confirmed requirements under ``keys`` (numeric inputs and categorical keys alike), for ``serves_requirements``."""
    out: list[str] = []
    for k in keys:
        if k in ctx.inputs:
            out.append(ctx.inputs[k].requirement.id)
            continue
        out += [r.id for r in ctx.ir.requirements.requirements
                if (canonical_key(r.key) or r.key) == k and r.value is not None and r.value.provenance.is_authoritative and r.id not in out]
    return out


class NetBook:
    """Net members collected while a block is built, declared at the end (a net may gather pins from several parts)."""

    def __init__(self) -> None:
        self._pins: dict[str, list[tuple[str, str]]] = {}
        self._kind: dict[str, NetKind] = {}
        self._note: dict[str, str] = {}
        self._serves: dict[str, list[str]] = {}

    def add(self, net: str, kind: NetKind, pins: Iterable[tuple[str, str]], note: str = "", serves: Iterable[str] = ()) -> None:
        have = self._kind.setdefault(net, kind)
        if have is not kind:
            raise TemplateRefusal(f"net {net!r} is declared {have.value} and {kind.value}")
        self._pins.setdefault(net, []).extend(pins)
        if note and net not in self._note:
            self._note[net] = note
        s = self._serves.setdefault(net, [])
        s += [x for x in serves if x not in s]

    def declare(self, b: BlockBuilder) -> None:
        for name, pins in self._pins.items():
            b.net(name, self._kind[name], pins, self._note.get(name, f"{name} net"), serves=self._serves.get(name, []))


def exclude(b: BlockBuilder, ref: str, reason: str) -> None:
    b.bind(ref, SpiceBinding(exclude=True, exclude_reason=reason, provenance=b.ctx.provenance(f"{ref} excluded from the design deck")))


def two_terminal(b: BlockBuilder, placed: PlacedPart, device: SpiceDevice, value: Traced, note: str) -> SpiceBinding:
    """The binding of a symmetric two-terminal part (R / L / C / V) at ``value``, pins in library order."""
    return SpiceBinding(device=device, value=value, pin_order=[placed.pin("1"), placed.pin("2")], provenance=b.ctx.provenance(f"{placed.ref}: {note}"))


def ac_sweep(b: BlockBuilder, aid: str, variation: Traced, points: Traced, fstart: Traced, fstop: Traced, note: str) -> AnalysisSpec:
    return AnalysisSpec(id=aid, kind=SpiceAnalysis.AC, params={"variation": variation, "points": points, "fstart": fstart, "fstop": fstop},
                        provenance=b.ctx.provenance(note))


def index_choices(b: BlockBuilder, prefix: str, count: int) -> dict[int, tuple[str, Traced]]:
    """Counting numbers 1..count the top-C calculators take to name one element (choices, so the calculators' inputs are traced)."""
    return {k: (f"{prefix}.idx.{k}", b.choice(f"{prefix}.idx.{k}", float(k), None,
                                            f"element index {k}: a counting number the top-C calculators take to name one element (not a design value)"))
            for k in range(1, count + 1)}


@dataclass(frozen=True)
class PortLoad:
    """What loads a top-C network's ports besides the port models (``calc.rf.resonator.top_c.ported_*``): the collector feed choke on the
    source port (inductance and Q, returned to AC ground) and the load the network sees (``r_load_eff``: the load port's resistance in parallel
    with the next stage's base divider; the load port's own key when nothing else is there). Keys and traced values."""

    l_port: tuple[str, Traced]
    q_port: tuple[str, Traced]
    r_load_eff: tuple[str, Traced]


@dataclass
class TopC:
    """One built top-C network: its members, their fixture bindings, the inductors, the traced inputs of its calculators and the net pins it adds."""

    network: str
    members: list[str]
    bindings: dict[str, SpiceBinding]
    inductors: list[str]
    #: calculator input ids (n, f0, bw, l, r_source, r_load) and their traced values
    ids: tuple[str, ...]
    values: tuple[Traced, ...]
    #: the pins the network puts on its input / output nets and on GND
    in_pins: list[tuple[str, str]] = field(default_factory=list)
    out_pins: list[tuple[str, str]] = field(default_factory=list)
    gnd_pins: list[tuple[str, str]] = field(default_factory=list)
    #: the loaded ports (``None``: a network between its two port models only)
    port: PortLoad | None = None

    def _ported(self) -> tuple[tuple[str, ...], tuple[Traced, ...]]:
        assert self.port is not None
        n, f0, bw, l, r_s, r_l = self.values
        i_n, i_f0, i_bw, i_l, i_rs, i_rl = self.ids
        p = self.port
        return ((i_n, i_f0, i_bw, i_l, i_rs, p.l_port[0], p.q_port[0], i_rl, p.r_load_eff[0]),
                (n, f0, bw, l, r_s, p.l_port[1], p.q_port[1], r_l, p.r_load_eff[1]))

    def s21(self, q_u: Traced, q_key: str, f: Traced, f_key: str) -> Traced:
        if self.port is not None:
            ids, vals = self._ported()
            return radio.top_c_ported_s21_db(*vals, q_u, f, (*ids, q_key, f_key))
        return radio.top_c_s21_db(*self.values, q_u, f, (*self.ids, q_key, f_key))

    def rel(self, q_u: Traced, q_key: str, f: Traced, f_key: str, f_ref: Traced, ref_key: str) -> Traced:
        if self.port is not None:
            ids, vals = self._ported()
            return radio.top_c_ported_rel_s21_db(*vals, q_u, f, f_ref, (*ids, q_key, f_key, ref_key))
        return radio.top_c_rel_s21_db(*self.values, q_u, f, f_ref, (*self.ids, q_key, f_key, ref_key))


def build_top_c(
    b: BlockBuilder,
    network: str,
    *,
    what: str,
    n: tuple[str, Traced],
    f0: tuple[str, Traced],
    bw: tuple[str, Traced],
    l: tuple[str, Traced],
    r_source: tuple[str, Traced],
    r_load: tuple[str, Traced],
    idx: dict[int, tuple[str, Traced]],
    cap_refs: list[str],
    ind_refs: list[str],
    nodes: list[str],
    ind_part: str,
    netbook: NetBook,
    cap_part: str = "cap_0402",
    port: PortLoad | None = None,
) -> TopC:
    """Place and value a Butterworth top-C network of ``n`` resonators (``calc.rf.resonator.top_c.*``) and declare its resonator nets.

    ``cap_refs`` in order: the input tap, then per resonator its shunt
    capacitor followed by the coupling capacitor to the next one, then the
    output tap (2n + 1 references); ``ind_refs`` the n resonator inductors;
    ``nodes`` the n resonator net names. The input / output tap pins are
    returned (the caller puts them on its nets); every value is a computed
    parameter ``<network>.c_tap_in`` / ``.c_tap_out`` / ``.c_couple.<i>`` /
    ``.c_shunt.<i>`` and the fixture binding of its part. With ``port`` the
    network sits between loaded ports (a collector choke on the source port,
    a base divider on the load port): the source tap absorbs the port's
    reactance (``<network>.port_r`` / ``.port_x``, ``.c_tap_reactive``) and
    :meth:`TopC.s21` / :meth:`TopC.rel` are the ported network's exact
    response (the caller makes the choke and the divider fixture members).
    """
    order = int(round(float(n[1].value)))
    if len(cap_refs) != 2 * order + 1 or len(ind_refs) != order or len(nodes) != order:
        raise TemplateRefusal(f"{network}: {order} resonators need {2 * order + 1} capacitor, {order} inductor references and {order} node names")
    ids = (n[0], f0[0], bw[0], l[0], r_source[0], r_load[0])
    values = (n[1], f0[1], bw[1], l[1], r_source[1], r_load[1])
    top = TopC(network=network, members=[], bindings={}, inductors=list(ind_refs), ids=ids, values=values, port=port)
    base_ids = (n[0], f0[0], bw[0], l[0])
    base_vals = (n[1], f0[1], bw[1], l[1])
    if port is None:
        src, load = r_source, r_load
        c_in = b.computed(f"{network}.c_tap_in", radio.top_c_c_tap(*base_vals, r_source[1], (*base_ids, r_source[0])))
    else:
        pids = (r_source[0], port.l_port[0], port.q_port[0], f0[0])
        pr = b.computed(f"{network}.port_r", radio.top_c_port_r(r_source[1], port.l_port[1], port.q_port[1], f0[1], pids))
        px = b.computed(f"{network}.port_x", radio.top_c_port_x(r_source[1], port.l_port[1], port.q_port[1], f0[1], pids))
        src, load = (f"{network}.port_r", pr), port.r_load_eff
        c_in = b.computed(f"{network}.c_tap_in", radio.top_c_c_tap_reactive(*base_vals, pr, px, (*base_ids, f"{network}.port_r", f"{network}.port_x")))
    c_out = b.computed(f"{network}.c_tap_out", radio.top_c_c_tap(*base_vals, load[1], (*base_ids, load[0])))
    shunts = [b.computed(f"{network}.c_shunt.{i}", radio.top_c_c_shunt(n[1], idx[i][1], f0[1], bw[1], l[1], src[1], load[1],
                                                                      (n[0], idx[i][0], f0[0], bw[0], l[0], src[0], load[0])))
              for i in range(1, order + 1)]
    couples = [b.computed(f"{network}.c_couple.{i}", radio.top_c_c_couple(n[1], idx[i][1], f0[1], bw[1], l[1], (n[0], idx[i][0], f0[0], bw[0], l[0])))
               for i in range(1, order)]
    caps: list[tuple[str, Traced, str]] = [(cap_refs[0], c_in, f"{what}: input tap capacitor")]
    k = 1
    for i in range(order):
        caps.append((cap_refs[k], shunts[i], f"{what}: resonator {i + 1} shunt capacitor"))
        k += 1
        if i < order - 1:
            caps.append((cap_refs[k], couples[i], f"{what}: coupling capacitor between resonators {i + 1} and {i + 2}"))
            k += 1
    caps.append((cap_refs[-1], c_out, f"{what}: output tap capacitor"))
    placed: dict[str, PlacedPart] = {}
    for ref, value, desc in caps:
        placed[ref] = b.part(cap_part, ref, part_value(value.value), desc)
        top.bindings[ref] = two_terminal(b, placed[ref], SpiceDevice.C, value, f"{network} fixture binding")
    for i, ref in enumerate(ind_refs):
        placed[ref] = b.part(ind_part, ref, part_value(l[1].value), f"{what}: resonator {i + 1} inductor")
        top.bindings[ref] = two_terminal(b, placed[ref], SpiceDevice.L, l[1], f"{network} fixture binding")
    top.members = [c[0] for c in caps] + list(ind_refs)
    for ref in top.members:
        exclude(b, ref, FIXTURE_ONLY.format(network=network))
    # nets: tap_in 1 = input net, 2 = node 1; shunt / inductor between node i and GND; couple between node i and i + 1; tap_out 1 = node n, 2 = output
    top.in_pins = [(cap_refs[0], placed[cap_refs[0]].pin("1"))]
    top.out_pins = [(cap_refs[-1], placed[cap_refs[-1]].pin("2"))]
    node_pins: list[list[tuple[str, str]]] = [[] for _ in range(order)]
    node_pins[0].append((cap_refs[0], placed[cap_refs[0]].pin("2")))
    node_pins[-1].append((cap_refs[-1], placed[cap_refs[-1]].pin("1")))
    k = 1
    for i in range(order):
        shunt = cap_refs[k]
        node_pins[i] += [(shunt, placed[shunt].pin("1")), (ind_refs[i], placed[ind_refs[i]].pin("1"))]
        top.gnd_pins += [(shunt, placed[shunt].pin("2")), (ind_refs[i], placed[ind_refs[i]].pin("2"))]
        k += 1
        if i < order - 1:
            cc = cap_refs[k]
            node_pins[i].append((cc, placed[cc].pin("1")))
            node_pins[i + 1].append((cc, placed[cc].pin("2")))
            k += 1
    for name, pins in zip(nodes, node_pins):
        netbook.add(name, NetKind.RF, pins, f"{what}: resonator node")
    return top


@dataclass
class BiasStage:
    """One divider-biased BFR92 stage: its transistor, the pins it puts on the base / collector nets, and its I_C nominal."""

    q: PlacedPart
    base_pins: list[tuple[str, str]]
    collector_pins: list[tuple[str, str]]
    ic_key: str
    ic: Traced


@dataclass
class BiasValues:
    """The bias choices of one kind of stage (keys and traced values): divider, emitter R / C, choke, V_BE, the link level."""

    rail_key: str
    rail: Traced
    r_b1: tuple[str, Traced]
    r_b2: tuple[str, Traced]
    r_e: tuple[str, Traced]
    c_e: tuple[str, Traced]
    l_choke: tuple[str, Traced]
    c_dec: tuple[str, Traced]
    v_be: tuple[str, Traced]
    link_v: Traced
    tol: Traced
    v_b: tuple[str, Traced] | None = None
    ic: tuple[str, Traced] | None = None


def bias_values(b: BlockBuilder, prefix: str, stage: str, what: str, *, r_b1: float, r_b2: float, r_e: float, c_e: float, l_choke: float,
                choke_note: str, common: dict[str, tuple[str, Traced]], rail_key: str, rail: Traced) -> BiasValues:
    """The choices of a stage kind ``<prefix>.<stage>.*`` and its computed V_B / I_C (``calc.divider.v_out`` / ``calc.rf.bjt_bias.ic``)."""
    k = f"{prefix}.{stage}"
    vals = BiasValues(
        rail_key=rail_key, rail=rail,
        r_b1=(f"{k}.r_b1", b.choice(f"{k}.r_b1", r_b1, "ohm", f"{what}: upper base-divider resistor (from the rail)")),
        r_b2=(f"{k}.r_b2", b.choice(f"{k}.r_b2", r_b2, "ohm", f"{what}: lower base-divider resistor (to GND)")),
        r_e=(f"{k}.r_e", b.choice(f"{k}.r_e", r_e, "ohm", f"{what}: emitter resistor (sets I_C with V_B and V_BE)")),
        c_e=(f"{k}.c_e", b.choice(f"{k}.c_e", c_e, "F", f"{what}: emitter bypass capacitor (RF ground at the emitter)")),
        l_choke=(f"{k}.l_choke", b.choice(f"{k}.l_choke", l_choke, "H", f"{what}: collector feed choke - {choke_note} [UNVERIFIED: its self-resonance]")),
        c_dec=common["c_dec"], v_be=common["v_be"], link_v=common["link_v"][1], tol=common["bias_tol"][1],
    )
    vals.v_b = (f"{k}.v_b", b.computed(f"{k}.v_b", voltage_divider_output(rail, vals.r_b1[1], vals.r_b2[1], (rail_key, vals.r_b1[0], vals.r_b2[0]))))
    vals.ic = (f"{k}.ic", b.computed(f"{k}.ic", radio.bjt_bias_ic(vals.v_b[1], vals.v_be[1], vals.r_e[1], (vals.v_b[0], vals.v_be[0], vals.r_e[0]))))
    return vals


def common_bias(b: BlockBuilder, prefix: str) -> dict[str, tuple[str, Traced]]:
    """The choices every BFR92 stage of a block shares: V_BE, the rail decoupling, the 0 ohm links' 0 V, the bias tolerance."""
    return {
        "v_be": (f"{prefix}.v_be", b.choice(f"{prefix}.v_be", 0.8, "V",
                                          "base-emitter voltage of a BFR92 at its bias current, for calc.rf.bjt_bias.ic (about what model.npn's default "
                                          "saturation current gives at 2-5 mA; the real part's is a datasheet fact [UNVERIFIED: NXP BFR92AW datasheet])")),
        "c_dec": (f"{prefix}.c_dec", b.choice(f"{prefix}.c_dec", 1e-9, "F", "rail decoupling capacitor at each stage's feed (after the 0 ohm link)")),
        "link_v": (f"{prefix}.link_v", b.choice(f"{prefix}.link_v", 0.0, "V",
                                              "a 0 ohm bias-current link is simulated as a 0 V source (an ideal ammeter): exact for a 0 ohm link; on the board it is "
                                              "the point where the lab lifts the link and inserts an ammeter")),
        "bias_tol": (f"{prefix}.bias_tol", b.choice(f"{prefix}.bias_tol", 0.15, None,
                                                  "relative tolerance of every I_C bias expectation (calc.rf.bjt_bias.ic ignores the base current and takes V_BE "
                                                  "as a constant; model.npn is a generic card)")),
    }


def bias_stage(
    b: BlockBuilder,
    vals: BiasValues,
    *,
    q_ref: str,
    refs: dict[str, str],
    nets: dict[str, str],
    what: str,
    exp_id: str,
    npn_text: Traced,
    netbook: NetBook,
    serves: Iterable[str] = (),
    part_resistor: str = "res_0402",
    part_cap: str = "cap_0402",
    part_choke: str = "ind_0603",
) -> BiasStage:
    """Place one BFR92 stage with its divider bias, emitter R // C, 0 ohm link + decoupling and collector choke; add its I_C expectation.

    ``refs``: ``r_b1``, ``r_b2``, ``r_e``, ``c_e``, ``link``, ``c_dec``,
    ``choke``; ``nets``: ``base``, ``emitter``, ``collector``, ``feed``
    (between the link and the choke). The base and collector nets are
    returned as pins (the caller adds the filters' taps).
    """
    q = b.part("rf_npn", q_ref, "BFR92", what, serves)
    card = npn_card()
    q_bind = card_binding(q, card, npn_text, b.ctx.provenance(f"{q_ref}: model.npn (generic, not a BFR92 model)"))
    b.bind(q_ref, q_bind)
    parts = {
        "r_b1": b.part(part_resistor, refs["r_b1"], part_value(vals.r_b1[1].value), f"{what}: upper base-divider resistor"),
        "r_b2": b.part(part_resistor, refs["r_b2"], part_value(vals.r_b2[1].value), f"{what}: lower base-divider resistor"),
        "r_e": b.part(part_resistor, refs["r_e"], part_value(vals.r_e[1].value), f"{what}: emitter resistor"),
        "c_e": b.part(part_cap, refs["c_e"], part_value(vals.c_e[1].value), f"{what}: emitter bypass capacitor"),
        "link": b.part(part_resistor, refs["link"], "0", f"{what}: 0 ohm bias-current link (lift it to measure I_C)"),
        "c_dec": b.part(part_cap, refs["c_dec"], part_value(vals.c_dec[1].value), f"{what}: feed decoupling capacitor"),
        "choke": b.part(part_choke, refs["choke"], part_value(vals.l_choke[1].value), f"{what}: collector feed choke"),
    }
    for name, device, value in (("r_b1", SpiceDevice.R, vals.r_b1[1]), ("r_b2", SpiceDevice.R, vals.r_b2[1]), ("r_e", SpiceDevice.R, vals.r_e[1]),
                                ("c_e", SpiceDevice.C, vals.c_e[1]), ("c_dec", SpiceDevice.C, vals.c_dec[1]), ("choke", SpiceDevice.L, vals.l_choke[1])):
        b.bind(refs[name], two_terminal(b, parts[name], device, value, "bias network at its value"))
    link = parts["link"]  # pin 1 at the rail, pin 2 at the feed: i(link) is the current from the rail into the stage
    b.bind(refs["link"], SpiceBinding(device=SpiceDevice.V, value=vals.link_v, pin_order=[link.pin("1"), link.pin("2")],
                                      provenance=b.ctx.provenance(f"{refs['link']}: a 0 ohm link simulated as a 0 V source (an ideal ammeter)")))
    rail = RAIL_NET
    netbook.add(rail, NetKind.POWER, [(refs["r_b1"], parts["r_b1"].pin("1")), (refs["link"], link.pin("1"))], "RX_5V rail")
    netbook.add(nets["feed"], NetKind.POWER, [(refs["link"], link.pin("2")), (refs["c_dec"], parts["c_dec"].pin("1")), (refs["choke"], parts["choke"].pin("1"))],
                f"{what}: collector feed after the 0 ohm link")
    netbook.add(nets["emitter"], NetKind.ANALOG, [(q_ref, q.pin("E")), (refs["r_e"], parts["r_e"].pin("1")), (refs["c_e"], parts["c_e"].pin("1"))],
                f"{what}: emitter")
    netbook.add(GROUND_NET, NetKind.GROUND, [(refs["r_b2"], parts["r_b2"].pin("2")), (refs["r_e"], parts["r_e"].pin("2")), (refs["c_e"], parts["c_e"].pin("2")),
                                            (refs["c_dec"], parts["c_dec"].pin("2"))])
    assert vals.ic is not None
    b.result.expectations.append(Expectation(
        id=exp_id, analysis_id=BIAS_ANALYSIS, vector=f"i({refs['link']})", reduce=Reduce.VALUE, nominal=vals.ic[1], tol_rel=vals.tol,
        provenance=b.ctx.provenance(f"{what}: collector current through the 0 ohm link {refs['link']} at the bias nominal (calc.rf.bjt_bias.ic)"),
    ))
    base = [(q_ref, q.pin("B")), (refs["r_b1"], parts["r_b1"].pin("2")), (refs["r_b2"], parts["r_b2"].pin("1"))]
    collector = [(q_ref, q.pin("C")), (refs["choke"], parts["choke"].pin("2"))]
    return BiasStage(q=q, base_pins=base, collector_pins=collector, ic_key=vals.ic[0], ic=vals.ic[1])


def bias_analysis(b: BlockBuilder) -> None:
    """The design deck's operating point the bias expectations read (identical in every block, so a composition keeps one)."""
    if not any(a.id == BIAS_ANALYSIS for a in b.result.analyses):
        b.result.analyses.append(AnalysisSpec(id=BIAS_ANALYSIS, kind=SpiceAnalysis.OP, provenance=b.ctx.provenance(BIAS_NOTE)))


def fixture_port(name: str, net: str, z0: Traced, direction: str = "bidir") -> RFPort:
    return RFPort(name=name, net=net, kind="port", z0_ohm=z0, direction=direction)  # type: ignore[arg-type]


def row(exp_id: str, quantity: str, drive: str, to: str, at: Traced, nominal: Traced, *, tol: Traced | None = None, bound: str | None = None,
        ref_at: Traced | None = None) -> RFExpectation:
    return RFExpectation(id=exp_id, quantity=quantity, drive=drive, to=to, at=at, ref_at=ref_at, nominal=nominal, tol_abs=tol, bound=bound)  # type: ignore[arg-type]


def probe(probe_id: str, drive: str, to: str, at: Traced, ref_at: Traced) -> RFProbe:
    return RFProbe(id=probe_id, quantity="rel_s21_db", drive=drive, to=to, at=at, ref_at=ref_at)


@dataclass
class RadioPlan:
    """The plan-level values the front-end blocks share (key, traced)."""

    f_c: tuple[str, Traced]
    if1: tuple[str, Traced]
    side: tuple[str, Traced]
    lo1: tuple[str, Traced]
    z0: tuple[str, Traced]


def radio_plan(b: BlockBuilder) -> RadioPlan:
    """``rf.f_c`` (the confirmed carrier), ``rf.if1``, ``rf.lo1_side``, ``rf.lo1`` (``calc.rf.superhet.lo``) and ``rf.z0`` (the system impedance)."""
    ctx = b.ctx
    carrier = ctx.inputs.get("carrier_frequency")
    if carrier is None:
        raise TemplateRefusal(f"block {b.result.block_id} needs the confirmed carrier_frequency (the channel of the KR 447 MHz raster)")
    f_c = copy_input(b, "rf.f_c", carrier.traced)
    if1 = plan_value(b, "rf.if1", lambda: b.choice("rf.if1", 21.4e6, "Hz", "first IF: a standard crystal-filter frequency (the IF back-end's ladder)"))
    side = plan_value(b, "rf.lo1_side", lambda: b.choice(
        "rf.lo1_side", -1.0, None,
        "LO1 side: -1 = low side (LO1 = f_c - IF1), so the image 404.76 MHz lies below the band; a high-side LO would put it at 490.36 MHz, in UHF TV "
        "[UNVERIFIED: 「대한민국 주파수 분배표」]"))
    lo1 = plan_value(b, "rf.lo1", lambda: b.computed("rf.lo1", radio.superhet_lo(f_c, if1, side, ("rf.f_c", "rf.if1", "rf.lo1_side"))))
    z0_in = ctx.inputs.get("system_impedance")
    if z0_in is not None:
        z0 = copy_input(b, "rf.z0", z0_in.traced)
    else:
        z0 = plan_value(b, "rf.z0", lambda: b.choice("rf.z0", 50.0, "ohm", "system impedance of the RF and IF ports (no system_impedance requirement stated)"))
    return RadioPlan(f_c=("rf.f_c", f_c), if1=("rf.if1", if1), side=("rf.lo1_side", side), lo1=("rf.lo1", lo1), z0=("rf.z0", z0))


def model_q(b: BlockBuilder, f_hz: float) -> tuple[str, Traced]:
    """The inductor-Q model value of the band that holds ``f_hz`` (``model.l_q.*``), added once."""
    return model_value(b, inductor_q_key(f_hz))


def model_value(b: BlockBuilder, mv: ModelValue | str) -> tuple[str, Traced]:
    """A model value, added to this block once (or the composing template's, when shared)."""
    key = mv if isinstance(mv, str) else mv.key
    have = b.result.params.get(key)
    if have is None:
        have = b.ctx.shared.get(key)
    if have is not None:
        if key not in b.result.model_keys:
            b.result.model_keys.append(key)
        return key, have
    return key, b.model(mv)


# --------------------------------------------------------------------------- the front end (under the can)


class RxFrontendBlock(Block):
    """``RX_RF`` -> ``fe_bpf2`` -> BFR92 LNA -> ``fe_bpf3`` -> ``MIX_RF``, all under one shield can (see the module docstring)."""

    id = FRONTEND_ID
    title = "RF front end: 2-pole BPF, BFR92 LNA, 3-pole BPF (under the shield can)"
    interface_nets = FRONTEND_INTERFACE

    def __init__(self, shield: str = "shield_103") -> None:
        #: the parts-table key of the can. The design names BMI-S-102 (SH601), but ``placement.rf_floorplan`` packs only 16 of this block's
        #: 25 parts into its 13.3 x 13.3 mm fence (less the 1 mm ring, 1 mm spacing; measured on the 10.0.6 footprints), so the default is the
        #: BMI-S-103 (23 x 23 mm inside), which holds all of them
        self.shield = shield

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        nb = NetBook()
        serves_fc = requirement_ids(ctx, "carrier_frequency")
        serves_z0 = requirement_ids(ctx, "system_impedance")
        serves_sens = requirement_ids(ctx, "rx_sensitivity")
        try:
            p = radio_plan(b)
            rail_key, rail = rail_level(b, RAIL_NET)
            npn_text = b.card(npn_card())
            q_uhf = model_q(b, float(p.f_c[1].value))
            lna_port = model_value(b, LNA_PORT)
            adex_port = model_value(b, ADEX_PORT)
            idx = index_choices(b, "fe", 3)
            tol = b.choice("fe.net_tol", 1.0, "dB", "tolerance of every exact-network fixture row of the front end (the netlist realises the designed network)")
            image = b.computed("fe.image", radio.superhet_image(p.f_c[1], p.lo1[1], (p.f_c[0], p.lo1[0])))
            half_if = b.computed("fe.half_if", radio.superhet_half_if(p.f_c[1], p.lo1[1], (p.f_c[0], p.lo1[0])))
            # ---- the LNA's bias values first: its base divider loads fe_bpf2 and its collector choke feeds fe_bpf3 (both are fixture members)
            common = common_bias(b, "fe")
            lna_vals = bias_values(b, "fe", "lna", "LNA Q1", r_b1=3300.0, r_b2=2200.0, r_e=220.0, c_e=100e-12, l_choke=100e-9,
                                   choke_note=("about 280 ohm at 447 MHz, 5.6 times the LNA's 50 ohm port model: it shunts the collector port, so it is a member "
                                               "of fe_bpf3's fixture and fe_bpf3's input tap absorbs its reactance"),
                                   common=common, rail_key=rail_key, rail=rail)
            lna_div = b.computed("fe.lna.r_div", parallel_resistance(lna_vals.r_b1[1], lna_vals.r_b2[1], (lna_vals.r_b1[0], lna_vals.r_b2[0])))
            r_eff2 = ("fe_bpf2.r_load_eff", b.computed("fe_bpf2.r_load_eff", parallel_resistance(lna_port[1], lna_div, (lna_port[0], "fe.lna.r_div"))))
            # the power share of fe_bpf2's load that reaches the transistor (the port model) rather than its divider: r_load_eff / R_port = R_div / (R_port + R_div)
            share2 = b.computed("fe_bpf2.load_share", voltage_divider_ratio(lna_port[1], lna_div, (lna_port[0], "fe.lna.r_div")))
            share2_db = b.computed("fe_bpf2.load_share_db", power_ratio_to_db(share2, ("fe_bpf2.load_share",)))
            # ---- fe_bpf2: 2-pole, 40 MHz, into the LNA's base with its divider
            n2 = ("fe.bpf2.n", b.choice("fe.bpf2.n", 2.0, None, "fe_bpf2: resonators of the band-pass before the LNA (2: little loss before the LNA)"))
            bw2 = ("fe.bpf2.bw", b.choice("fe.bpf2.bw", 40e6, "Hz", "fe_bpf2: bandwidth (Butterworth; wide for low loss before the LNA)"))
            l2 = ("fe.bpf2.l", b.choice("fe.bpf2.l", 4.7e-9, "H", "fe_bpf2: resonator inductance (0604HQ; the taps must transform down: R_p = Q_e w0 L above the ports)"))
            bpf2 = build_top_c(b, "fe_bpf2", what="fe_bpf2 (2-pole band-pass before the LNA)", n=n2, f0=p.f_c, bw=bw2, l=l2, r_source=p.z0, r_load=r_eff2, idx=idx,
                               cap_refs=["C1", "C2", "C3", "C4", "C5"], ind_refs=["L1", "L2"], nodes=["FE2_R1", "FE2_R2"], ind_part="ind_0604hq", netbook=nb)
            # ---- the LNA
            bias_analysis(b)
            lna = bias_stage(b, lna_vals, q_ref="Q1", refs={"r_b1": "R1", "r_b2": "R2", "r_e": "R3", "c_e": "C6", "link": "R4", "c_dec": "C7", "choke": "L3"},
                             nets={"emitter": "LNA_E", "feed": "LNA_VC"}, what="LNA Q1 (BFR92, class A)", exp_id="ic_lna", npn_text=npn_text,
                             netbook=nb, serves=[*serves_fc, *serves_sens], part_choke="ind_0603")
            # ---- fe_bpf3: 3-pole, 20 MHz, from the LNA's collector port with its feed choke
            n3 = ("fe.bpf3.n", b.choice("fe.bpf3.n", 3.0, None, "fe_bpf3: resonators of the band-pass after the LNA (image and LO1 rejection)"))
            bw3 = ("fe.bpf3.bw", b.choice("fe.bpf3.bw", 20e6, "Hz", "fe_bpf3: bandwidth (Butterworth; the discrete proposal's 437.8-457.9 MHz)"))
            l3 = ("fe.bpf3.l", b.choice("fe.bpf3.l", 4.7e-9, "H", "fe_bpf3: resonator inductance (0604HQ)"))
            port3 = PortLoad(l_port=lna_vals.l_choke, q_port=q_uhf, r_load_eff=adex_port)
            bpf3 = build_top_c(b, "fe_bpf3", what="fe_bpf3 (3-pole band-pass after the LNA)", n=n3, f0=p.f_c, bw=bw3, l=l3, r_source=lna_port, r_load=adex_port, idx=idx,
                               cap_refs=["C8", "C9", "C10", "C11", "C12", "C13", "C14"], ind_refs=["L4", "L5", "L6"], nodes=["FE3_R1", "FE3_R2", "FE3_R3"],
                               ind_part="ind_0604hq", netbook=nb, port=port3)
            # ---- fixture nominals (the exact networks) and the Cohn losses the budget uses
            nom: dict[str, Traced] = {}
            # fe_bpf2's own S21 is into r_load_eff; the fixture reads it into the port model (the transistor), so its row adds the load share
            s21_net2 = b.computed("fe_bpf2.s21_net", bpf2.s21(q_uhf[1], q_uhf[0], p.f_c[1], p.f_c[0]))
            nom["fe_bpf2.s21"] = b.computed("fe_bpf2.s21", radio.db_sum(s21_net2, share2_db, ("fe_bpf2.s21_net", "fe_bpf2.load_share_db")))
            nom["fe_bpf3.s21"] = b.computed("fe_bpf3.s21", bpf3.s21(q_uhf[1], q_uhf[0], p.f_c[1], p.f_c[0]))
            for top, name in ((bpf2, "fe_bpf2"), (bpf3, "fe_bpf3")):
                nom[f"{name}.rel_image"] = b.computed(f"{name}.rel_image", top.rel(q_uhf[1], q_uhf[0], image, "fe.image", p.f_c[1], p.f_c[0]))
                n_t, bw_t = (n2, bw2) if name == "fe_bpf2" else (n3, bw3)
                b.computed(f"{name}.loss", radio.bpf_dissipation_loss(n_t[1], p.f_c[1], bw_t[1], q_uhf[1], (n_t[0], p.f_c[0], bw_t[0], q_uhf[0])))
            nom["fe_bpf3.rel_lo1"] = b.computed("fe_bpf3.rel_lo1", bpf3.rel(q_uhf[1], q_uhf[0], p.lo1[1], p.lo1[0], p.f_c[1], p.f_c[0]))
            # ---- sweep
            lin = b.choice("fe.sweep.lin", "lin", None, "linear ac sweep variation")
            pts = b.choice("fe.sweep.points", 601, None, "points of the front-end filters' sweep (0.5 MHz apart; rows are read at their own point analyses, the sweep is the figure)")
            f_lo = b.choice("fe.sweep.fstart", 300e6, "Hz", "start of the front-end filters' sweep")
            f_hi = b.choice("fe.sweep.fstop", 600e6, "Hz", "stop of the front-end filters' sweep")
            sweep = ac_sweep(b, "fe_sweep", lin, pts, f_lo, f_hi, "front-end filters: 300-600 MHz")
        except ValueError as e:
            raise TemplateRefusal(f"block {self.id}: {e}") from e
        # ---- shield can, nets
        sh = b.part(self.shield, "SH1", SHIELD_VALUE.get(self.shield, "RFShield"), "shield can over the LNA and both band-pass filters")
        exclude(b, "SH1", "shield can: no electrical model (its fence pads are GND)")
        nb.add(GROUND_NET, NetKind.GROUND, [*bpf2.gnd_pins, *bpf3.gnd_pins, *sh.at("Shield")])
        nb.add("RX_RF", NetKind.RF, bpf2.in_pins, "RF input: the bench U.FL (stage 3) or the T/R switch's RX port (transceiver)", serves=[*serves_fc, *serves_z0])
        nb.add("LNA_IN", NetKind.RF, [*bpf2.out_pins, *lna.base_pins], "LNA input: fe_bpf2's output tap, the base and its divider")
        nb.add("LNA_OUT", NetKind.RF, [*lna.collector_pins, *bpf3.in_pins], "LNA output: the collector, its choke and fe_bpf3's input tap")
        nb.add("MIX_RF", NetKind.RF, bpf3.out_pins, "the mixer's RF input: fe_bpf3's output tap")
        nb.declare(b)
        # ---- fixtures
        # the LO-spur responses are probes (recorded, no verdict) when the LO chain, built first, shared them
        spur_probes = [(pid, ctx.shared[f"lo.{pid}"]) for pid in LO_SPUR_RESPONSES if f"lo.{pid}" in ctx.shared]
        # every part on the filters' port nets is a member: fe_bpf2 holds the LNA's base divider R1 / R2 (to RX_5V and GND), fe_bpf3 the LNA's
        # collector choke L3 with its 0 ohm link R4 and feed decoupling C7 (the link returns the choke to RX_5V: the fixture's rail port, an ac short)
        rail_port = RFPort(name="rx_5v", net=RAIL_NET, kind="rail", voltage_v=rail, direction="in")
        for top, a, net_a, z_a, c, net_c, z_c, extra, extra_q in (
                (bpf2, "rx_rf", "RX_RF", p.z0, "lna_in", "LNA_IN", lna_port, ["R1", "R2"], {}),
                (bpf3, "lna_out", "LNA_OUT", lna_port, "mix_rf", "MIX_RF", adex_port, ["L3", "R4", "C7"], {"L3": q_uhf[1]})):
            name = top.network
            exps = [row("s21_fc", "s21_db", a, c, p.f_c[1], nom[f"{name}.s21"], tol=tol),
                    row("rel_image", "rel_s21_db", a, c, image, nom[f"{name}.rel_image"], tol=tol, ref_at=p.f_c[1])]
            if name == "fe_bpf3":
                exps.append(row("rel_lo1", "rel_s21_db", a, c, p.lo1[1], nom["fe_bpf3.rel_lo1"], tol=tol, ref_at=p.f_c[1]))
            prb = [probe("half_if", a, c, half_if, p.f_c[1]), *[probe(pid, a, c, f, p.f_c[1]) for pid, f in spur_probes]]
            b.result.networks.append(RFNetwork(
                id=name, block=self.id, members=[*top.members, *extra], bindings=top.bindings, loss_q={**{r: q_uhf[1] for r in top.inductors}, **extra_q},
                q_ref_hz=p.f_c[1], ports=[fixture_port(a, net_a, z_a[1], "in"), fixture_port(c, net_c, z_c[1], "out"), rail_port], sweep=[sweep],
                expectations=exps, probes=prb,
            ))
        # ---- block ports, chain, plan, lab
        b.result.ports = [
            RFPort(name="RX_RF", net="RX_RF", kind="port", z0_ohm=p.z0[1], frequency_hz=p.f_c[1], direction="in"),
            RFPort(name="MIX_RF", net="MIX_RF", kind="port", z0_ohm=adex_port[1], frequency_hz=p.f_c[1], direction="out"),
            RFPort(name="RX_5V", net=RAIL_NET, kind="rail", voltage_v=rail, direction="in"),
        ]
        b.result.chain = ["C1", "L1", "C2", "C3", "L2", "C4", "C5", "Q1", "L3", "C8", "L4", "C9", "C10", "L5", "C11", "C12", "L6", "C13", "C14"]
        b.result.shield_ref = "SH1"
        b.result.plan_lines += [
            PlanLine(id="image", kind="response", f_hz=image, ref_hz=p.f_c[1],
                     points_to=["spice.rf.fe_bpf2.rel_image", "spice.rf.fe_bpf3.rel_image", "rf.lab.image_rejection"],
                     note="the image LO1 - IF1 of the low-side LO: rejected only by fe_bpf2 + fe_bpf3 (about 47 dB at Q_u 40)"),
            PlanLine(id="half_if", kind="response", f_hz=half_if, ref_hz=p.f_c[1], points_to=["spice.rf.fe_bpf3", "rf.lab.half_if"],
                     note="the 2x2 response (f + LO1) / 2: the LC front end gives a few dB; the ADEX-10's 2x2 suppression carries it [UNVERIFIED]"),
        ]
        b.result.lab_items += [
            LabItem(id="lna", block=self.id, what="LNA gain, noise figure and input / output match (the BFR92's S-parameters)",
                    instruments=["VNA", "noise-figure meter"], reason="no S-parameters or noise model of the BFR92 in this project: the LNA ports are model values"),
            LabItem(id="image_rejection", block=self.id, what="image (LO1 - IF1) response rejection of the receiver, and blocking / IP3",
                    instruments=["two signal generators", "SINAD meter"], reason="the fixtures judge the filters under model values; the receiver's rejection is measured"),
            LabItem(id="half_if", block=self.id, what="half-IF (2x2) response rejection at (f_c + LO1) / 2",
                    instruments=["signal generator", "SINAD meter"], reason="rests on the ADEX-10's 2x2 suppression, a datasheet fact [UNVERIFIED]"),
        ]
        b.result.notes.append(f"{self.id}: fe_bpf2 / fe_bpf3 nominals are the exact top-C networks (calc.rf.resonator.top_c.*) at Q_u {q_uhf[1].value:g}; "
                              f"a fixture PASS is a network verdict under confirmed model values, never the LNA's gain or NF")
        return b.done()


# --------------------------------------------------------------------------- mixer, diplexer, IF1 post-amp


class RxMixerBlock(Block):
    """ADEX-10 first mixer -> diplexer -> BFR92 IF1 post-amplifier -> ``IF1`` (see the module docstring); also the receiver's sensitivity budget."""

    id = MIXER_ID
    title = "first mixer ADEX-10, IF diplexer, BFR92 IF1 post-amplifier"
    interface_nets = MIXER_INTERFACE

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        nb = NetBook()
        serves_fc = requirement_ids(ctx, "carrier_frequency")
        serves_z0 = requirement_ids(ctx, "system_impedance")
        serves_build = requirement_ids(ctx, "radio_build", "modulation")
        sens_req = ctx.inputs.get("rx_sensitivity")
        try:
            p = radio_plan(b)
            rail_key, rail = rail_level(b, RAIL_NET)
            npn_text = b.card(npn_card())
            adex_port = model_value(b, ADEX_PORT)
            ifamp_port = model_value(b, IFAMP_PORT)
            q_if = model_q(b, float(p.if1[1].value))
            k_p1 = b.choice("fe.order.p1", 1.0, None, "order +1 (the sum: f + x)")
            f_sum = b.computed("fe.sum_product", radio.mult_spur(p.lo1[1], p.f_c[1], k_p1, (p.lo1[0], p.f_c[0], "fe.order.p1")))
            # ---- diplexer
            l_dip = b.choice("fe.dip.l", 470e-9, "H", "diplexer: series inductor to the post-amp (0603; about 63 ohm at IF1, 1.26 kohm at LO1)")
            c_ser = b.computed("fe.dip.c_series", lc_c_for_resonance(p.if1[1], l_dip, (p.if1[0], "fe.dip.l")))
            c_hp = b.choice("fe.dip.c_hp", 22e-12, "F", "diplexer: capacitor of the absorptive branch (about 17 ohm at LO1, 340 ohm at IF1)")
            r_term = b.choice("fe.dip.r_term", 50.0, "ohm", "diplexer: absorptive termination of the LO / RF / sum products (the mixer's IF port wants a broadband 50 ohm)")
            s21_min = b.choice("fe.dip.s21_min", -1.0, "dB", "diplexer: least S21 at IF1 (passband loss target)")
            s11_max = b.choice("fe.dip.s11_max", -10.0, "dB", "diplexer: largest S11 at the mixer's IF port at LO1 and at the sum product (the products see a termination)")
            dlin = b.choice("fe.dip.sweep.dec", "dec", None, "logarithmic ac sweep variation")
            dpts = b.choice("fe.dip.sweep.points", 50, None, "points per decade of the diplexer's sweep")
            dlo = b.choice("fe.dip.sweep.fstart", 1e6, "Hz", "start of the diplexer's sweep")
            dhi = b.choice("fe.dip.sweep.fstop", 2e9, "Hz", "stop of the diplexer's sweep")
            dsweep = ac_sweep(b, "dip_sweep", dlin, dpts, dlo, dhi, "diplexer: 1 MHz - 2 GHz")
            # ---- post-amp
            common = common_bias(b, "fe")
            ifa_vals = bias_values(b, "fe", "ifa", "IF1 post-amp Q50", r_b1=3300.0, r_b2=2200.0, r_e=220.0, c_e=10e-9, l_choke=2.2e-6,
                                   choke_note="about 296 ohm at IF1, 5.9 times the post-amp's 50 ohm load", common=common, rail_key=rail_key, rail=rail)
            c_out = b.choice("fe.ifa.c_out", 10e-9, "F", "post-amp output DC block to IF1 (0.74 ohm at IF1)")
            # ---- sensitivity budget (Friis from the back; passive losses add in dB)
            loss2, loss3 = b.ctx.shared.get("fe_bpf2.loss"), b.ctx.shared.get("fe_bpf3.loss")
            if loss2 is None or loss3 is None:
                raise TemplateRefusal(f"block {self.id}: the sensitivity budget needs the front-end filters' losses fe_bpf2.loss / fe_bpf3.loss "
                                      "(build rx_frontend first and share its parameters)")
            nf_ifb, nf_ifa, g_ifa = model_value(b, IFB_NF), model_value(b, IFAMP_NF), model_value(b, IFAMP_GAIN)
            cl, nf_lna, g_lna = model_value(b, ADEX_CL), model_value(b, LNA_NF), model_value(b, LNA_GAIN)
            nf_a = b.computed("fe.nf.after_mixer", friis_nf(nf_ifa[1], g_ifa[1], nf_ifb[1], (nf_ifa[0], g_ifa[0], nf_ifb[0])))
            nf_b = b.computed("fe.nf.at_mixer", radio.db_sum(cl[1], nf_a, (cl[0], "fe.nf.after_mixer")))
            nf_c = b.computed("fe.nf.at_bpf3", radio.db_sum(loss3, nf_b, ("fe_bpf3.loss", "fe.nf.at_mixer")))
            nf_d = b.computed("fe.nf.at_lna", friis_nf(nf_lna[1], g_lna[1], nf_c, (nf_lna[0], g_lna[0], "fe.nf.at_bpf3")))
            nf = b.computed("fe.nf.total", radio.db_sum(loss2, nf_d, ("fe_bpf2.loss", "fe.nf.at_lna")))
            t0 = b.choice("fe.t0", 290.0, "K", "noise reference temperature T0 (IEEE: 290 K)")
            if_bw = plan_value(b, "rf.if_bw", lambda: b.choice("rf.if_bw", 7.5e3, "Hz", "IF noise bandwidth: the ladder's -3 dB bandwidth, 0.6 x the 12.5 kHz channel spacing"))
            snr = b.choice("fe.snr", 11.0, "dB", "C/N at the 12 dB SINAD point of an FM discriminator (an assumed number, not measured)")
            sens = b.computed("fe.sensitivity", sensitivity(t0, if_bw, nf, snr, ("fe.t0", "rf.if_bw", "fe.nf.total", "fe.snr")))
        except ValueError as e:
            raise TemplateRefusal(f"block {self.id}: {e}") from e
        # ---- parts
        u = b.part("mixer", "U50", "ADEX-10", "first mixer (passive double-balanced, +7 dBm LO)", [*serves_fc, *serves_build])
        exclude(b, "U50", "ADEX-10 has no SPICE model: excluded; its ports are model.adex10.port_r in the fixtures")
        dl = b.part("ind_0603", "L50", part_value(l_dip.value), "diplexer: series inductor to the post-amp")
        dc = b.part("cap_0402", "C50", part_value(c_ser.value), "diplexer: series capacitor, resonant with the inductor at IF1")
        dh = b.part("cap_0402", "C51", part_value(c_hp.value), "diplexer: absorptive branch capacitor")
        dr = b.part("res_0402", "R50", part_value(r_term.value), "diplexer: absorptive termination")
        dip_bind = {
            "L50": two_terminal(b, dl, SpiceDevice.L, l_dip, "diplexer fixture binding"),
            "C50": two_terminal(b, dc, SpiceDevice.C, c_ser, "diplexer fixture binding"),
            "C51": two_terminal(b, dh, SpiceDevice.C, c_hp, "diplexer fixture binding"),
            "R50": two_terminal(b, dr, SpiceDevice.R, r_term, "diplexer fixture binding"),
        }
        for ref in dip_bind:
            exclude(b, ref, FIXTURE_ONLY.format(network="diplexer"))
        bias_analysis(b)
        ifa = bias_stage(b, ifa_vals, q_ref="Q50", refs={"r_b1": "R51", "r_b2": "R52", "r_e": "R53", "c_e": "C52", "link": "R54", "c_dec": "C53", "choke": "L51"},
                         nets={"emitter": "IFA2_E", "feed": "IFA2_VC"}, what="IF1 post-amp Q50 (BFR92, class A)", exp_id="ic_ifamp", npn_text=npn_text,
                         netbook=nb)
        co = b.part("cap_0402", "C54", part_value(c_out.value), "post-amp output DC block to IF1")
        b.bind("C54", two_terminal(b, co, SpiceDevice.C, c_out, "output DC block at its value"))
        nb.add("MIX_RF", NetKind.RF, u.at("RF"), "the mixer's RF input")
        nb.add("LO1_MIX", NetKind.RF, u.at("LO"), "the mixer's LO input (from the LO chain's pad)")
        nb.add("MIX_IF", NetKind.RF, [*u.at("IF"), ("L50", dl.pin("1")), ("C51", dh.pin("1"))], "the mixer's IF output into the diplexer")
        nb.add("DIP_M", NetKind.RF, [("L50", dl.pin("2")), ("C50", dc.pin("1"))], "diplexer: between the series inductor and capacitor")
        nb.add("DIP_T", NetKind.RF, [("C51", dh.pin("2")), ("R50", dr.pin("1"))], "diplexer: absorptive branch")
        nb.add("IFA2_B", NetKind.RF, [("C50", dc.pin("2")), *ifa.base_pins], "post-amp input: the diplexer's output, the base and its divider")
        nb.add("IFA2_C", NetKind.RF, [*ifa.collector_pins, ("C54", co.pin("1"))], "post-amp collector")
        nb.add("IF1", NetKind.RF, [("C54", co.pin("2"))], "IF1 output (21.4 MHz, 50 ohm) to the IF back-end", serves=serves_z0)
        nb.add(GROUND_NET, NetKind.GROUND, [*u.at("GND"), ("R50", dr.pin("2"))])
        nb.declare(b)
        # ---- the diplexer fixture: its load is the post-amp's base with the base divider R51 / R52 beside it (members; R51 returns to RX_5V,
        # the fixture's rail port - an ac short), so the rows see the divider the board hangs on IFA2_B
        b.result.networks.append(RFNetwork(
            id="diplexer", block=self.id, members=["L50", "C50", "C51", "R50", "R51", "R52"], bindings=dip_bind, loss_q={"L50": q_if[1]}, q_ref_hz=p.if1[1],
            ports=[fixture_port("mix_if", "MIX_IF", adex_port[1], "in"), fixture_port("ifa_in", "IFA2_B", ifamp_port[1], "out"),
                   RFPort(name="rx_5v", net=RAIL_NET, kind="rail", voltage_v=rail, direction="in")], sweep=[dsweep],
            expectations=[
                row("s21_if1", "s21_db", "mix_if", "ifa_in", p.if1[1], s21_min, bound="at_least"),
                row("s11_lo1", "s11_db", "mix_if", "mix_if", p.lo1[1], s11_max, bound="at_most"),
                row("s11_sum", "s11_db", "mix_if", "mix_if", f_sum, s11_max, bound="at_most"),
            ],
        ))
        b.result.ports = [
            RFPort(name="MIX_RF", net="MIX_RF", kind="port", z0_ohm=adex_port[1], frequency_hz=p.f_c[1], direction="in"),
            RFPort(name="LO1_MIX", net="LO1_MIX", kind="port", z0_ohm=adex_port[1], frequency_hz=p.lo1[1], direction="in"),
            RFPort(name="IF1", net="IF1", kind="port", z0_ohm=p.z0[1], frequency_hz=p.if1[1], direction="out"),
            RFPort(name="RX_5V", net=RAIL_NET, kind="rail", voltage_v=rail, direction="in"),
        ]
        b.result.chain = ["U50", "L50", "C50", "C51", "R50", "Q50", "L51", "C54"]
        stated = ""
        if sens_req is not None:
            stated = (f"; the stated requirement {sens_req.requirement.id} rx_sensitivity = {float(sens_req.traced.value):.4g} dBm is compared with the estimate "
                      f"{float(sens.value):.4g} dBm only by this lab item: an estimate is never a verdict (decision 2A)")
        b.result.lab_items += [
            LabItem(id="mixer", block=self.id, what="ADEX-10 conversion loss, LO drive level and port match",
                    instruments=["signal generator", "spectrum analyser", "power meter"], reason="the ADEX-10 has no model: every mixer number is a datasheet fact [UNVERIFIED]"),
            LabItem(id="sensitivity", block=self.id,
                    what=(f"12 dB SINAD sensitivity of the receiver (stage 3 + stage 2 by cable); the budget estimate is {float(sens.value):.4g} dBm at NF "
                          f"{float(nf.value):.3g} dB (Friis under model values){stated}"),
                    instruments=["signal generator", "SINAD meter", "noise-figure meter"],
                    reason="the noise figures, gains and the conversion loss are UNVERIFIED model values; only a measurement says what the receiver hears"),
            LabItem(id="lo_radiation", block=self.id, what="LO1 leakage out of the RF input (the receiver's spurious emission at LO1)",
                    instruments=["spectrum analyser"], reason="LO-to-RF isolation of the mixer and the LNA's reverse isolation are not modelled [UNVERIFIED]"),
            LabItem(id="ifamp_output", block=self.id,
                    what=("the IF1 post-amplifier's output impedance at IF1 against the system impedance it drives (the bench cable, or the transceiver's "
                          "IF1 filter input match): the IF1 port declares the system impedance, but the BFR92 collector with its feed choke is not matched "
                          "to it"),
                    instruments=["VNA"],
                    reason=("no output match is designed and no S-parameters of the BFR92 exist here: the IF1 filter's fixture drives its input from the "
                            "declared system impedance, which the post-amp is only assumed to present [UNVERIFIED]")),
        ]
        return b.done()


__all__ = [
    "ADEX_PORT",
    "BIAS_ANALYSIS",
    "FRONTEND_ID",
    "FRONTEND_INTERFACE",
    "FRONTEND_NETWORKS",
    "LNA_PORT",
    "MIXER_ID",
    "MIXER_INTERFACE",
    "MIXER_NETWORKS",
    "RAIL_KEYS",
    "RAIL_NET",
    "RxFrontendBlock",
    "RxMixerBlock",
    "PortLoad",
    "build_top_c",
    "copy_input",
    "plan_value",
    "radio_plan",
    "rail_level",
]
