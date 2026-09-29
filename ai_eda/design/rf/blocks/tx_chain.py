"""The transmit chain of the KR 447 MHz FM radio (kr447 design §2.4, decision 1B, part P12): TCXO -> two buffered PM tanks -> x3 -> x2 -> x2 + BPF -> driver.

Invariant: every part comes from the kr447 parts table by library name
(:mod:`ai_eda.design.rf.parts`), every number is a registered calculator's
output over the blocks' confirmed choices, the ``model.*`` values
(UNVERIFIED, listed in ``ir.rf.model_values``) and the confirmed
requirements, and every passive network between the excluded or unmodelled
parts is an RF fixture (``ir.rf.networks``) whose rows the fixture runner
judges on ngspice. Nothing here says that the modulator deviates, that the
multipliers multiply or that the driver drives: the TCXO and the PHA-1 are
excluded, a class-C harmonic generator is not modelled (``model.npn`` is a
generic card at an operating point), and a fixture PASS is "a network verdict
under confirmed model values (not a measured part)" at schematic level - no
track, via or ground-return inductance.

The frequency plan (decision 1B): N = ``rf.n_mult`` = 12 = 3 x 2 x 2
(``tx.mult.1`` .. ``.3``, refused unless their product is N and the last
stage lands on the carrier), the reference f_T = f_c / N
(``calc.clock.divided``: 37.296875 MHz for the 447.5625 MHz channel, inside
the KT2520K-T library range "10-60MHz" - a custom frequency whose
availability and stability are lab / procurement items), the stage outputs
f_1 / f_2 / f_3 = 3 / 6 / 12 f_T (``calc.rf.mult.stage``: 111.89 / 223.78 /
447.5625 MHz).

Three blocks, because a shield can holds every part of its block
(``placement.rf_floorplan``) and the design's one BMI-S-105 can over the
modulator *and* the multipliers (SH801) cannot: its fence, less the placer's
1 mm ring, is 34.9 x 22.2 mm, and the shelf packer (1 mm between courtyards)
fitted 75 of the 82 parts both blocks had before the output band-pass moved
into the chain (91 now), whatever the order - so the modulator has its own
can:

* :class:`TxModBlock` (``tx_mod``, local references 1..13 and ``C_T1`` /
  ``C_T2``, ``SH2``, ``TP1``; re-based to 8xx) - under the can ``SH2``
  (``shield_103``, a BMI-S-103; ``shield=None`` for none): the TCXO ``Y1``
  (``TX_3V3``) -> ``pm_mod1`` -> the MMBT3904 emitter follower ``Q1`` ->
  ``pm_mod2`` -> the follower ``Q2``, whose emitter is the interface
  ``TX_PM_OUT`` (the modulated reference at f_T); the varactor bias divider,
  its feed / bypass and the audio coupling from ``PM_DRIVE``.
* :class:`TxChainBlock` (``tx_chain``, local references 12..39, ``SH1``,
  ``TP2`` / ``TP3``; re-based to 8xx) - under the can ``SH1`` (the
  BMI-S-105 of the design's SH801): ``TX_PM_OUT`` through ``C12`` -> three
  BFR92 stages ``Q3`` .. ``Q5`` (divider bias, emitter R // C, 0 ohm link +
  decoupling, collector choke); the x3 and x6 collectors each drive a
  double-tuned (2-resonator) top-C tank ``tx_tank1`` / ``tx_tank2`` with end
  external Q ``tx.tank_qe`` = 20 (bandwidth
  ``calc.rf.resonator.top_c.bw_for_qe``) into the next stage's base, the x12
  collector the 5-resonator output band-pass ``tx_bpf`` at f_c straight into
  ``TX_RAW`` - the driver pad's matched input ``tx.z_mid`` (the design's last
  tank and its 3-pole band-pass designed as one network: two filters joined
  tap to tap have no resistive node between them, and the pair's two
  fixtures, each between its own port models, claimed 13 dB more rejection
  at f_c + f_T than the cascade gives). Build it after :class:`TxModBlock` on
  one board (it reads the plan values the modulator wrote from
  :attr:`BlockContext.shared`).
* :class:`TxDriverBlock` (``tx_driver``, local references 50..99, re-based
  to 8xx): the matched pi pad ``drv_pad`` (``TX_RAW`` -> ``DRV_IN``), the
  PHA-1 driver ``U50`` (DC blocks in and out, bias through the choke ``L53``
  from ``TX_5V`` into RF_OUT) and the matched pi pad ``pa_pad`` into
  ``PA_IN`` - the PA's input (block ``pa``). ``tx.pa_pad.a_db`` sets the PA
  drive; re-choosing it after the conducted measurement is a human design
  change.

**The phase modulator** (two buffered tanks on one bias node, kr447 design
§2.4 1B row). Each tank is a source port behind the DC block ``pm.c_dc`` and
the series resistor R_s into the tank node: C_fixed + the trimmer at mid
(``pm.c_trim``) + the varactor (``model.varactor.*``, ``DVAR``) to ground and
L (``calc.rf.lc.l_for_resonance`` at f_T with ``pm.c_tot``) returned to the
bias node ``VAR_B`` (bypass ``pm.c_bypass``, fed through ``pm.r_feed`` from
the divider node ``PM_BIAS``, which the audio ``PM_DRIVE`` reaches through
``pm.c_audio``). R_s = ``calc.rf.pm.source_r_loaded`` makes the tank's loaded
Q ``pm.q_l`` with the port and the next follower's input
``model.buf.r_in`` as its load; the follower is AC-coupled (``pm.c_buf``)
with its own divider bias. The design had one follower between the tanks and
"light coupling (choice)" into the x3 stage: a second follower ``Q2`` replaces
that coupling, because the only load ``calc.rf.pm.tank_phase_loaded`` models
is a resistance at the tank node - so both tanks are the same designed
network (between ``model.tcxo.r_out`` / ``model.buf.r_out`` and
``model.buf.r_in``, 50 ohm / 10 kohm by default) and their fixtures measure
exactly what the calculator computes.

``pm_mod1`` / ``pm_mod2``: three states ``bias_lo`` / ``bias_nom`` /
``bias_hi`` on the control port at ``PM_BIAS`` (``pm.v_lo`` / ``pm.v_bias``
/ ``pm.v_hi``: V0 -/0/+ the varactor's peak swing at 300 Hz and full
deviation), each with one ``phase21_deg`` row at f_T whose nominal is the
exact network phase ``calc.rf.pm.tank_phase_loaded`` (with the varactor's
C(V) from ``calc.rf.varactor.c_at_bias``), tol_abs ``pm.phase_tol`` 1 deg -
the ideal-tank formula is 0.8-1.4 deg off this network (kr447 design §2.4
critic2). ``rf.deviation`` takes each tank's chord slope from the recorded
bias_lo / bias_hi phases and sums the two. The integrator of the TX audio
block is designed with ``tx.k_pm`` - a choice equal to twice the per-tank
small-signal ``calc.rf.pm.k_pm`` (no registered calculator adds two slopes);
the check that matters, ``rf.deviation``, multiplies the measured slopes, not
this design number.

The audio path ``PM_DRIVE`` -> ``VAR_B`` (the factor ``a`` of
``rf.deviation``) is checked in the design deck: ``pm_couple_300`` /
``_1k`` / ``_3k`` (DB_AT of v(VAR_B) re v(PM_DRIVE) on single-point ac
analyses) against the 1 kHz nominal - the high-pass of ``pm.c_audio`` with
the divider's Thevenin resistance and the low-pass of ``pm.r_feed`` with
``pm.c_bypass`` (``calc.audio.highpass1.db_at`` + ``.lowpass1.db_at``,
``calc.rf.db_sum``), tol_abs ``pm.couple.tol_ref`` at 1 kHz and
``pm.couple.tol_band`` at 300 Hz / 3 kHz: the rows hold only when both corners
stay outside the voice band (the low-impedance bias feed of critic2; the
bypass stays large, because it also sets the tank's centre offset).

**The multiplier tanks and the final filter.** Every network is designed
between *loaded* ports, and every part on its port nets is a member of its
fixture: the source is the collector port model ``model.bfr92.r_out``
(1 kohm) in parallel with the stage's collector choke (its Q at the stage
frequency, returned to AC ground through the 0 ohm link, the fixture's rail
port ``tx_5v`` on ``TX_5V``), whose reactance the input tap absorbs
(``calc.rf.resonator.top_c.port_r`` / ``.port_x`` /
``.c_tap_reactive``); the load of a tank is the next stage's port model
``model.bfr92.r_in`` (500 ohm) in parallel with that stage's base divider
(``tx.<next>.r_div``, ``tx.tank<k>.r_load_eff``), both dividers' resistors
members. The tank inductances are choices (``tx.tank1.l`` 150 nH,
``.tank2.l`` 68 nH; the band-pass ``tx.bpf.l`` 27 nH) such that the
capacitive taps can transform (a tap only transforms down: R_p = Q_e w0 L
must exceed the loaded collector port; the design's 50 ohm example values
are refused there, kr447 design §2.7 critic2 (c)). Every tank row's nominal
is the exact ported network (``calc.rf.resonator.top_c.ported_s21_db`` at
f_k; ``.ported_rel_s21_db`` at f_k -/+ f_T, ``calc.rf.mult.spur``), tol_abs
``tx.net_tol`` 1 dB - the design named Cohn's
``calc.rf.bpf.dissipation_loss`` (4.34 dB, the resonators' dissipation
alone) for the s21 row; it is recorded beside the nominal
(``tx_tank<k>.loss``). At the default choices (Q_u 40) the tanks pass
-5.50 / -5.54 dB and reject - / + f_T by 47.84 / 26.24 dB (x3, 111.89 MHz)
and 29.01 / 17.67 dB (x2, 223.78 MHz); the + side is the weak one. The
tanks carry no trimmers (Murata TZB4-A footprints would not fit the
BMI-S-105 fence with the stages): alignment is by part selection, a lab
item.

``tx_bpf`` (5 resonators ``tx.bpf.n``, ``tx.bpf.bw`` 25 MHz, from the x12
collector with its choke into the pad's ``tx.z_mid``) has the design's
one-sided rows: s21 at f_c at least ``tx.bpf.s21_min`` (-13 dB; the network
gives -12.36 dB at Q_u 40), f_c -/+ f_T at most the exact rejection +
``tx.bpf.margin`` (the network gives 42.32 / 31.35 dB, bounded at 41.32 /
30.35 dB), f_c / 2 at most -50 dB and 2 f_c at most -20 dB; f_c -/+ 2 f_T,
-/+ 3 f_T and 3/2 f_c are probes (recorded). The pads
(``calc.rf.attenuator.pi.*``) are judged s21 = -A (through
``calc.divider.ratio`` and ``calc.rf.voltage_ratio_to_db``), tol
``tx.pad.tol``, and s11 at most ``tx.pad.s11_max``.

The design deck is the five transistors' bias (``tx_ic_buf1``,
``tx_ic_buf2``, ``tx_ic_x3``, ``tx_ic_x6``, ``tx_ic_x12`` on ``op_bias``:
I_C through each 0 ohm link simulated as a 0 V source, at the nominal of
``calc.rf.bjt_bias.ic``, tol_rel ``tx.bias_tol``) and the three
``pm_couple_*`` rows. Every tank / filter part is a fixture member only (the
deck leaves it out; its fixture binds it at its calculator value).

Each block names the signal-integrity class of its nets in its
``net_classes`` (:data:`MOD_NET_CLASSES`, :data:`CHAIN_NET_CLASSES`,
:data:`DRIVER_NET_CLASSES`; re-based with the block's net prefix): ``RF50``
for the 50 ohm lines from ``TX_RAW`` through the pads and the driver to the
PA input, ``TX_LUMPED`` for the lumped tank, resonator and follower nodes (no line impedance, never marked RF, so
``domain.rf.impedance`` judges only the lines). The composing template
declares the classes.

Plan values shared with other blocks (``rf.f_c``, ``rf.z0``, ``rf.n_mult``
and the ``tx.*`` plan of :func:`tx_plan`) are taken from
:attr:`BlockContext.shared` when a composing template wrote them, else written
here (``rf.n_mult`` with the family's one row of :mod:`ai_eda.design.rf.common`,
so every block that declares it gives the same row); the rails' levels ``power.tx_5v`` / ``power.tx_3v3`` must be
shared by the supply block (the power block of part P9, or a bench
stand-in).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from ai_eda.ir import AnalysisSpec, Expectation, NetKind, Reduce, SpiceBinding, SpiceDevice, Traced
from ai_eda.ir.rf import LabItem, PlanLine, RFExpectation, RFNetwork, RFPort, RFProbe, RFState
from ai_eda.tools.calc import radio
from ai_eda.tools.calc.basic import clock_divided, divider_r1_for_v_out, parallel_resistance, rc_time_constant, voltage_divider_output, voltage_divider_ratio
from ai_eda.tools.calc.part_value import format_part_value
from ai_eda.tools.calc.rf import lc_l_for_resonance, ppm_offset, voltage_ratio_to_db
from ai_eda.tools.spice import SpiceAnalysis

from ai_eda.design.inputs import canonical_key
from ai_eda.design.library_parts import TemplateRefusal
from ai_eda.design.rf.blocks.base import GROUND_NET, Block, BlockBuilder, BlockContext, BlockResult
from ai_eda.design.rf.common import N_MULT, N_MULT_KEY, N_MULT_TEXT
from ai_eda.design.rf.models import ModelCard, ModelValue, card_binding, inductor_q_key, npn_card, varactor_card
from ai_eda.design.rf.parts import PlacedPart

#: block ids (``ir.rf.blocks`` / the networks' ``block``)
MOD_ID = "tx_mod"
CHAIN_ID = "tx_chain"
DRIVER_ID = "tx_driver"
#: the modulated reference between the modulator and the multiplier chain (the second follower's emitter)
PM_OUT_NET = "TX_PM_OUT"
#: the interface nets each block keeps (never prefixed)
MOD_INTERFACE: tuple[str, ...] = ("TX_5V", "TX_3V3", "PM_DRIVE", PM_OUT_NET)
CHAIN_INTERFACE: tuple[str, ...] = ("TX_5V", PM_OUT_NET, "TX_RAW")
DRIVER_INTERFACE: tuple[str, ...] = ("TX_RAW", "TX_5V", "PA_IN")
#: the fixture networks each block declares
MOD_NETWORKS: tuple[str, ...] = ("pm_mod1", "pm_mod2")
CHAIN_NETWORKS: tuple[str, ...] = ("tx_tank1", "tx_tank2", "tx_bpf")
DRIVER_NETWORKS: tuple[str, ...] = ("drv_pad", "pa_pad")
#: the multiplier of each stage (decision 1B: 3 x 2 x 2 = 12)
STAGE_MULTIPLIERS: tuple[int, ...] = (3, 2, 2)
#: the default tank inductances (H) of the x3 and x6 stages' tanks: R_p = Q_e w0 L above the collector port with its choke (module docstring)
TANK_L: tuple[float, ...] = (150e-9, 68e-9)
#: the default collector chokes of the three stages (H)
STAGE_CHOKES: tuple[float, ...] = (4.7e-6, 1e-6, 470e-9)
#: the rail nets and the supply block's parameter that holds each level (read from BlockContext.shared)
RAIL_KEYS: dict[str, str] = {"TX_5V": "power.tx_5v", "TX_3V3": "power.tx_3v3"}
#: the design-deck analysis every bias expectation reads (the id and note of the receive chain's bias analysis: a transceiver keeps one)
BIAS_ANALYSIS = "op_bias"
BIAS_NOTE = "operating point: the RF transistors' bias (I_C through the 0 ohm links)"
#: the multiplication factor and the row text the TX audio block (part P9) declares it with: identical rows merge as one
#: the integrator's design modulator constant (the TX audio block's ``k_pm_key`` on this board)
K_PM_KEY = "tx.k_pm"
#: the phase-modulator states and their parameter keys
PM_STATES: tuple[tuple[str, str, str], ...] = (("bias_lo", "pm.v_lo", "lo"), ("bias_nom", "pm.v_bias", "nom"), ("bias_hi", "pm.v_hi", "hi"))
#: the design-deck rows of the PM_DRIVE -> varactor factor ``a`` (``rf.deviation`` reads them by these ids)
PM_COUPLE_IDS: tuple[str, ...] = ("pm_couple_300", "pm_couple_1k", "pm_couple_3k")
#: the design-deck bias rows
BIAS_IDS: tuple[str, ...] = ("tx_ic_buf1", "tx_ic_buf2", "tx_ic_x3", "tx_ic_x6", "tx_ic_x12")
#: the value text of each shield-can row (the Laird part number)
SHIELD_VALUE: dict[str, str] = {"shield_102": "BMI-S-102", "shield_103": "BMI-S-103", "shield_105": "BMI-S-105"}
#: the signal-integrity class of each block's nets (``BlockResult.net_classes``, local names; the composing template declares the classes):
#: ``RF50`` the 50 ohm lines from ``TX_RAW`` to the PA input, ``TX_LUMPED`` the lumped tank / resonator / follower nodes (no line impedance, not marked RF)
MOD_NET_CLASSES: dict[str, tuple[str, ...]] = {
    "TX_LUMPED": ("TCXO_OUT", "PM1_A", "PM1_T", "PM2_A", "PM2_T", "BUF1_B", "BUF1_E", "BUF2_B", PM_OUT_NET),
}
CHAIN_NET_CLASSES: dict[str, tuple[str, ...]] = {
    "TX_LUMPED": ("TX_X3_B", "TX_X3_C", "TX_X6_B", "TX_X6_C", "TX_X12_B", "TX_X12_C", "TX_T1_R1", "TX_T1_R2", "TX_T2_R1", "TX_T2_R2",
                  "TX_B_R1", "TX_B_R2", "TX_B_R3", "TX_B_R4", "TX_B_R5"),
    "RF50": ("TX_RAW",),
}
DRIVER_NET_CLASSES: dict[str, tuple[str, ...]] = {
    "RF50": ("TX_RAW", "DRV_IN", "DRV_RFIN", "DRV_OUT", "PA_PAD_IN", "PA_IN"),
}

#: the model values this module adds (port resistances of excluded parts)
DRIVER_PORT = ModelValue("model.driver.port_r", 50.0, "ohm", "input and output resistance of the PHA-1 TX driver (excluded from every netlist)",
                         "Mini-Circuits PHA-1 datasheet")
PA_INPUT = ModelValue("model.pa.r_in", 50.0, "ohm", "RF input resistance of the MMZ09332BT1 PA (excluded from every netlist; its input match is internal)",
                      "NXP MMZ09332B datasheet")

#: why a network part is not in the design deck
FIXTURE_ONLY = ("an RF network part: simulated only in its block's RF fixture ({network}) at its calculator value - the design deck judges only the "
                "transistors' bias and the audio path to the varactors")


# --------------------------------------------------------------------------- shared helpers (the pa block imports these)


def part_value(x: float) -> str:
    """A part value as KiCad writes it (``4.7n``, ``3.3k``)."""
    return format_part_value(float(x))


def plan_value(b: BlockBuilder, key: str, make: Callable[[], Traced]) -> Traced:
    """A plan-level value: the composing template's (``ctx.shared``) when it wrote one, else ``make()`` (which writes it in this block)."""
    have = b.result.params.get(key)
    if have is not None:
        return have
    have = b.ctx.shared.get(key)
    return have if have is not None else make()


def copy_input(b: BlockBuilder, key: str, traced: Traced) -> Traced:
    """A confirmed requirement copied into a parameter (``design.inputs_vs_requirements`` re-reads it), unless a composing block already did."""
    return plan_value(b, key, lambda: b._param(key, traced))


def rail_level(b: BlockBuilder, net: str) -> tuple[str, Traced]:
    """``(key, level)`` of a rail the supply block (or its bench stand-in) writes; refuses when the composition did not share it."""
    key = RAIL_KEYS[net]
    t = b.ctx.shared.get(key)
    if t is None:
        raise TemplateRefusal(f"block {b.result.block_id}: the level of {net} is the supply block's parameter {key!r} (the power block's, or a bench "
                              "stand-in's), which the composing template did not share (BlockContext.shared): compose the supply block first")
    return key, t


def requirement_ids(ctx: BlockContext, *keys: str) -> list[str]:
    """The ids of the confirmed requirements under ``keys`` (numeric inputs and categorical keys alike), for ``serves_requirements``."""
    out: list[str] = []
    for k in keys:
        if k in ctx.inputs:
            if ctx.inputs[k].requirement.id not in out:
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


def passive(b: BlockBuilder, key: str, ref: str, value: Traced, description: str, *, deck: bool = True, network: str = "",
            serves: Iterable[str] = ()) -> tuple[PlacedPart, SpiceBinding]:
    """A resistor / capacitor / inductor of the parts table at ``value`` (the device by the unit): its binding, bound in the deck or excluded (a fixture member)."""
    device = {"ohm": SpiceDevice.R, "F": SpiceDevice.C, "H": SpiceDevice.L}.get(value.unit or "")
    if device is None:
        raise TemplateRefusal(f"{ref}: a passive's value must be in ohm, F or H, got {value.unit!r}")
    placed = b.part(key, ref, part_value(value.value), description, serves)
    binding = two_terminal(b, placed, device, value, f"at its value{' (' + network + ' fixture binding)' if network else ''}")
    if deck:
        b.bind(ref, binding)
    else:
        exclude(b, ref, FIXTURE_ONLY.format(network=network))
    return b.result.placed[ref], binding


def ac_sweep(b: BlockBuilder, aid: str, variation: Traced, points: Traced, fstart: Traced, fstop: Traced, note: str) -> AnalysisSpec:
    return AnalysisSpec(id=aid, kind=SpiceAnalysis.AC, params={"variation": variation, "points": points, "fstart": fstart, "fstop": fstop},
                        provenance=b.ctx.provenance(note))


def index_choices(b: BlockBuilder, prefix: str, count: int) -> dict[int, tuple[str, Traced]]:
    """Counting numbers 1..count the calculators take to name one element (choices, so their inputs are traced)."""
    return {k: (f"{prefix}.idx.{k}", plan_value(b, f"{prefix}.idx.{k}", lambda k=k: b.choice(
        f"{prefix}.idx.{k}", float(k), None, f"element index {k}: a counting number the filter calculators take to name one element (not a design value)")))
            for k in range(1, count + 1)}


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


def model_q(b: BlockBuilder, f_hz: float) -> tuple[str, Traced]:
    """The inductor-Q model value of the band that holds ``f_hz`` (``model.l_q.*``), added once."""
    return model_value(b, inductor_q_key(f_hz))


def fixture_port(name: str, net: str, z0: Traced, direction: str = "bidir") -> RFPort:
    return RFPort(name=name, net=net, kind="port", z0_ohm=z0, direction=direction)  # type: ignore[arg-type]


def row(exp_id: str, quantity: str, drive: str, to: str, at: Traced, nominal: Traced, *, tol: Traced | None = None, bound: str | None = None,
        ref_at: Traced | None = None, state: str | None = None) -> RFExpectation:
    return RFExpectation(id=exp_id, state=state, quantity=quantity, drive=drive, to=to, at=at, ref_at=ref_at, nominal=nominal, tol_abs=tol, bound=bound)  # type: ignore[arg-type]


def probe(probe_id: str, drive: str, to: str, at: Traced, ref_at: Traced) -> RFProbe:
    return RFProbe(id=probe_id, quantity="rel_s21_db", drive=drive, to=to, at=at, ref_at=ref_at)


def calc(b: BlockBuilder, key: str, make: Callable[[], Traced]) -> Traced:
    """A calculator output as a computed parameter (a refusal of the calculator refuses the block with its sentence)."""
    have = b.result.params.get(key)
    if have is not None:
        return have
    try:
        traced = make()
    except (ValueError, ArithmeticError) as e:
        raise TemplateRefusal(f"block {b.result.block_id}: {key}: {e}") from e
    return b.computed(key, traced)


# --------------------------------------------------------------------------- the top-C networks (tanks and the final band-pass)


@dataclass(frozen=True)
class PortLoad:
    """What loads a top-C network's ports besides the port models (``calc.rf.resonator.top_c.ported_*``): the collector feed choke on the
    source port (its inductance and Q, returned to AC ground) and the load the network sees - the load port's resistance in parallel with the
    next stage's base divider (``r_load_eff``; the load port's own key when nothing else is there). Keys and traced values."""

    l_port: tuple[str, Traced]
    q_port: tuple[str, Traced]
    r_load_eff: tuple[str, Traced]


@dataclass
class TopC:
    """One built top-C network: its members, their fixture bindings, the inductors, the calculator inputs and the net pins it adds."""

    network: str
    members: list[str]
    bindings: dict[str, SpiceBinding]
    inductors: list[str]
    #: calculator input ids (n, f0, bw, l, r_source, r_load) and their traced values
    ids: tuple[str, ...]
    values: tuple[Traced, ...]
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


def build_top_c(b: BlockBuilder, network: str, *, what: str, n: tuple[str, Traced], f0: tuple[str, Traced], bw: tuple[str, Traced], l: tuple[str, Traced],
                r_source: tuple[str, Traced], r_load: tuple[str, Traced], idx: dict[int, tuple[str, Traced]], cap_refs: list[str], ind_refs: list[str],
                nodes: list[str], ind_part: str, netbook: NetBook, cap_part: str = "cap_0402", port: PortLoad | None = None) -> TopC:
    """Place and value a Butterworth top-C network of ``n`` resonators (``calc.rf.resonator.top_c.*``) and declare its resonator nets.

    ``cap_refs`` in order: the input tap, then per resonator its shunt
    capacitor followed by the coupling capacitor to the next one, then the
    output tap (2n + 1 references); ``ind_refs`` the n resonator inductors
    (returned to GND); ``nodes`` the n resonator net names. Every value is a
    computed parameter ``<network>.c_tap_in`` / ``.c_tap_out`` /
    ``.c_couple.<i>`` / ``.c_shunt.<i>`` and the fixture binding of its part;
    every part is excluded from the design deck.

    ``port``: the network sits between *loaded* ports (a collector choke on
    the source port, the next stage's base divider on the load port): the
    source tap absorbs the port's reactance (``<network>.port_r`` /
    ``.port_x``, ``.c_tap_reactive``), the shunts and the load tap are
    designed for ``R'`` and ``r_load_eff``, and :meth:`TopC.s21` /
    :meth:`TopC.rel` are the ported network's exact response - the choke and
    the divider are then members of the network's fixture.
    """
    order = int(round(float(n[1].value)))
    if len(cap_refs) != 2 * order + 1 or len(ind_refs) != order or len(nodes) != order:
        raise TemplateRefusal(f"{network}: {order} resonators need {2 * order + 1} capacitor, {order} inductor references and {order} node names")
    ids = (n[0], f0[0], bw[0], l[0], r_source[0], r_load[0])
    values = (n[1], f0[1], bw[1], l[1], r_source[1], r_load[1])
    top = TopC(network=network, members=[], bindings={}, inductors=list(ind_refs), ids=ids, values=values, port=port)
    base_ids, base_vals = (n[0], f0[0], bw[0], l[0]), (n[1], f0[1], bw[1], l[1])
    if port is None:
        src, load = r_source, r_load
        c_in = calc(b, f"{network}.c_tap_in", lambda: radio.top_c_c_tap(*base_vals, r_source[1], (*base_ids, r_source[0])))
    else:
        pids = (r_source[0], port.l_port[0], port.q_port[0], f0[0])
        pr = calc(b, f"{network}.port_r", lambda: radio.top_c_port_r(r_source[1], port.l_port[1], port.q_port[1], f0[1], pids))
        px = calc(b, f"{network}.port_x", lambda: radio.top_c_port_x(r_source[1], port.l_port[1], port.q_port[1], f0[1], pids))
        src, load = (f"{network}.port_r", pr), port.r_load_eff
        c_in = calc(b, f"{network}.c_tap_in", lambda: radio.top_c_c_tap_reactive(*base_vals, pr, px, (*base_ids, f"{network}.port_r", f"{network}.port_x")))
    c_out = calc(b, f"{network}.c_tap_out", lambda: radio.top_c_c_tap(*base_vals, load[1], (*base_ids, load[0])))
    shunts = [calc(b, f"{network}.c_shunt.{i}", lambda i=i: radio.top_c_c_shunt(n[1], idx[i][1], f0[1], bw[1], l[1], src[1], load[1],
                                                                                (n[0], idx[i][0], f0[0], bw[0], l[0], src[0], load[0])))
              for i in range(1, order + 1)]
    couples = [calc(b, f"{network}.c_couple.{i}", lambda i=i: radio.top_c_c_couple(n[1], idx[i][1], f0[1], bw[1], l[1], (n[0], idx[i][0], f0[0], bw[0], l[0])))
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
        placed[ref], top.bindings[ref] = passive(b, cap_part, ref, value, desc, deck=False, network=network)
    for i, ref in enumerate(ind_refs):
        placed[ref], top.bindings[ref] = passive(b, ind_part, ref, l[1], f"{what}: resonator {i + 1} inductor", deck=False, network=network)
    top.members = [c[0] for c in caps] + list(ind_refs)
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
        netbook.add(name, NetKind.ANALOG, pins, f"{what}: resonator node (a high-impedance lumped node, no line impedance)")
    return top


# --------------------------------------------------------------------------- transistor stages


@dataclass
class BiasValues:
    """The bias choices of one stage (keys and traced values) and its computed V_B / I_C."""

    rail_key: str
    rail: Traced
    r_b1: tuple[str, Traced]
    r_b2: tuple[str, Traced]
    r_e: tuple[str, Traced]
    c_e: tuple[str, Traced] | None
    l_choke: tuple[str, Traced] | None
    common: dict[str, tuple[str, Traced]]
    v_b: tuple[str, Traced] | None = None
    ic: tuple[str, Traced] | None = None


def common_bias(b: BlockBuilder) -> dict[str, tuple[str, Traced]]:
    """The choices every transistor stage of the TX chain shares: V_BE, the feed decoupling, the 0 ohm links' 0 V, the bias tolerance."""
    c = lambda key, value, unit, text: (key, plan_value(b, key, lambda: b.choice(key, value, unit, text)))  # noqa: E731
    return {
        "v_be": c("tx.v_be", 0.8, "V", ("base-emitter voltage of the BFR92 / MMBT3904 stages at their bias current, for calc.rf.bjt_bias.ic (about what "
                                        "model.npn's default saturation current gives at 1-3 mA; the real parts' are datasheet facts [UNVERIFIED: NXP BFR92AW, "
                                        "MMBT3904 datasheets])")),
        "c_dec": c("tx.c_dec", 1e-9, "F", "feed decoupling capacitor of each transistor stage (after its 0 ohm link)"),
        "link_v": c("tx.link_v", 0.0, "V", ("a 0 ohm bias-current link is simulated as a 0 V source (an ideal ammeter): exact for a 0 ohm link; on the board "
                                            "it is the point where the lab lifts the link and inserts an ammeter")),
        "bias_tol": c("tx.bias_tol", 0.15, None, ("relative tolerance of every I_C bias expectation (calc.rf.bjt_bias.ic ignores the base current and takes "
                                                  "V_BE as a constant; model.npn is a generic card)")),
    }


def bias_values(b: BlockBuilder, stage: str, what: str, *, r_b1: float, r_b2: float, r_e: float, c_e: float | None, l_choke: float | None,
                choke_note: str, common: dict[str, tuple[str, Traced]], rail_key: str, rail: Traced) -> BiasValues:
    """The choices of one stage ``tx.<stage>.*`` and its computed V_B / I_C (``calc.divider.v_out`` / ``calc.rf.bjt_bias.ic``)."""
    k = f"tx.{stage}"
    vals = BiasValues(
        rail_key=rail_key, rail=rail,
        r_b1=(f"{k}.r_b1", b.choice(f"{k}.r_b1", r_b1, "ohm", f"{what}: upper base-divider resistor (from TX_5V)")),
        r_b2=(f"{k}.r_b2", b.choice(f"{k}.r_b2", r_b2, "ohm", f"{what}: lower base-divider resistor (to GND)")),
        r_e=(f"{k}.r_e", b.choice(f"{k}.r_e", r_e, "ohm", f"{what}: emitter resistor (sets I_C with V_B and V_BE)")),
        c_e=None if c_e is None else (f"{k}.c_e", b.choice(f"{k}.c_e", c_e, "F", f"{what}: emitter bypass capacitor (RF ground at the emitter)")),
        l_choke=None if l_choke is None else (f"{k}.l_choke", b.choice(f"{k}.l_choke", l_choke, "H",
                                                                        f"{what}: collector feed choke - {choke_note} [UNVERIFIED: its self-resonance]")),
        common=common,
    )
    vals.v_b = (f"{k}.v_b", calc(b, f"{k}.v_b", lambda: voltage_divider_output(rail, vals.r_b1[1], vals.r_b2[1], (rail_key, vals.r_b1[0], vals.r_b2[0]))))
    v_be = common["v_be"]
    vals.ic = (f"{k}.ic", calc(b, f"{k}.ic", lambda: radio.bjt_bias_ic(vals.v_b[1], v_be[1], vals.r_e[1], (vals.v_b[0], v_be[0], vals.r_e[0]))))  # type: ignore[index]
    return vals


@dataclass
class Stage:
    """One placed transistor stage: the transistor and the pins it puts on its base / emitter / collector nets."""

    q: PlacedPart
    base_pins: list[tuple[str, str]]
    emitter_pins: list[tuple[str, str]]
    collector_pins: list[tuple[str, str]]


def transistor_stage(b: BlockBuilder, vals: BiasValues, *, part: str, value: str, q_ref: str, refs: dict[str, str], nets: dict[str, str], what: str,
                     exp_id: str, npn_text: Traced, netbook: NetBook, serves: Iterable[str] = ()) -> Stage:
    """Place one divider-biased stage: the transistor, R_b1 / R_b2, R_E (// C_E when given), the 0 ohm link + decoupling, the choke when given.

    ``refs``: ``r_b1``, ``r_b2``, ``r_e``, ``link``, ``c_dec`` and optionally
    ``c_e``, ``choke``; ``nets``: ``emitter`` (declared here; the caller may
    add pins to it) and ``feed`` (after the link); with a choke the returned
    collector pins are the transistor's and the choke's far end (the caller
    adds the tank's input tap and names the net). Without a choke the collector sits on the feed
    (an emitter follower: the emitter is the output). Adds the I_C
    expectation ``exp_id`` (the link's current) on :data:`BIAS_ANALYSIS`.
    """
    q = b.part(part, q_ref, value, what, serves)
    card = npn_card()
    b.bind(q_ref, card_binding(q, card, npn_text, b.ctx.provenance(f"{q_ref}: model.npn (generic, not a {value} model)")))
    common = vals.common
    rb1, _ = passive(b, "res_0402", refs["r_b1"], vals.r_b1[1], f"{what}: upper base-divider resistor")
    rb2, _ = passive(b, "res_0402", refs["r_b2"], vals.r_b2[1], f"{what}: lower base-divider resistor")
    re_, _ = passive(b, "res_0402", refs["r_e"], vals.r_e[1], f"{what}: emitter resistor")
    cd, _ = passive(b, "cap_0402", refs["c_dec"], common["c_dec"][1], f"{what}: feed decoupling capacitor")
    link = b.part("res_0402", refs["link"], "0", f"{what}: 0 ohm bias-current link (lift it to measure I_C)")
    b.bind(refs["link"], SpiceBinding(device=SpiceDevice.V, value=common["link_v"][1], pin_order=[link.pin("1"), link.pin("2")],
                                      provenance=b.ctx.provenance(f"{refs['link']}: a 0 ohm link simulated as a 0 V source (an ideal ammeter)")))
    netbook.add("TX_5V", NetKind.POWER, [(refs["r_b1"], rb1.pin("1")), (refs["link"], link.pin("1"))], "TX_5V rail")
    netbook.add(nets["feed"], NetKind.POWER, [(refs["link"], link.pin("2")), (refs["c_dec"], cd.pin("1"))], f"{what}: supply feed after the 0 ohm link")
    emitter = [(q_ref, q.pin("E")), (refs["r_e"], re_.pin("1"))]
    gnd = [(refs["r_b2"], rb2.pin("2")), (refs["r_e"], re_.pin("2")), (refs["c_dec"], cd.pin("2"))]
    if vals.c_e is not None:
        ce, _ = passive(b, "cap_0402", refs["c_e"], vals.c_e[1], f"{what}: emitter bypass capacitor")
        emitter.append((refs["c_e"], ce.pin("1")))
        gnd.append((refs["c_e"], ce.pin("2")))
    netbook.add(GROUND_NET, NetKind.GROUND, gnd)
    netbook.add(nets["emitter"], NetKind.ANALOG, emitter, f"{what}: emitter")
    if vals.l_choke is not None:
        ch, _ = passive(b, "ind_0603", refs["choke"], vals.l_choke[1], f"{what}: collector feed choke")
        netbook.add(nets["feed"], NetKind.POWER, [(refs["choke"], ch.pin("1"))])
        collector = [(q_ref, q.pin("C")), (refs["choke"], ch.pin("2"))]
    else:
        netbook.add(nets["feed"], NetKind.POWER, [(q_ref, q.pin("C"))])
        collector = []
    assert vals.ic is not None
    b.result.expectations.append(Expectation(
        id=exp_id, analysis_id=BIAS_ANALYSIS, vector=f"i({refs['link']})", reduce=Reduce.VALUE, nominal=vals.ic[1], tol_rel=common["bias_tol"][1],
        provenance=b.ctx.provenance(f"{what}: collector current through the 0 ohm link {refs['link']} at the bias nominal (calc.rf.bjt_bias.ic)"),
    ))
    base = [(q_ref, q.pin("B")), (refs["r_b1"], rb1.pin("2")), (refs["r_b2"], rb2.pin("1"))]
    return Stage(q=q, base_pins=base, emitter_pins=emitter, collector_pins=collector)


def bias_analysis(b: BlockBuilder) -> None:
    """The design deck's operating point the bias expectations read (identical in every block, so a composition keeps one)."""
    if not any(a.id == BIAS_ANALYSIS for a in b.result.analyses):
        b.result.analyses.append(AnalysisSpec(id=BIAS_ANALYSIS, kind=SpiceAnalysis.OP, provenance=b.ctx.provenance(BIAS_NOTE)))


# --------------------------------------------------------------------------- the frequency plan


@dataclass
class TxPlan:
    """The TX frequency plan (key, traced): f_c, the system impedance, N, the stage multipliers, f_T, the stage outputs, the spur orders."""

    f_c: tuple[str, Traced]
    z0: tuple[str, Traced]
    n_mult: tuple[str, Traced]
    mults: list[tuple[str, Traced]]
    f_ref: tuple[str, Traced]
    stages: list[tuple[str, Traced]]
    orders: dict[str, tuple[str, Traced]]
    z_mid: tuple[str, Traced]
    net_tol: Traced


def tx_plan(b: BlockBuilder) -> TxPlan:
    """``rf.f_c``, ``rf.z0``, ``rf.n_mult``, ``tx.mult.<k>``, ``tx.f_ref`` (f_c / N), ``tx.f<k>``, the orders, ``tx.z_mid`` - shared first."""
    ctx = b.ctx
    carrier = ctx.inputs.get("carrier_frequency")
    if carrier is None:
        raise TemplateRefusal(f"block {b.result.block_id} needs the confirmed carrier_frequency (the channel of the KR 447 MHz raster)")
    f_c = copy_input(b, "rf.f_c", carrier.traced)
    z0_in = ctx.inputs.get("system_impedance")
    if z0_in is not None:
        z0 = copy_input(b, "rf.z0", z0_in.traced)
    else:
        z0 = plan_value(b, "rf.z0", lambda: b.choice("rf.z0", 50.0, "ohm", "system impedance of the RF and IF ports (no system_impedance requirement stated)"))
    n = plan_value(b, N_MULT_KEY, lambda: b.choice(N_MULT_KEY, N_MULT, None, N_MULT_TEXT))  # the family's one row (ai_eda.design.rf.common)
    mults = []
    for k, m in enumerate(STAGE_MULTIPLIERS, start=1):
        key = f"tx.mult.{k}"
        mults.append((key, plan_value(b, key, lambda k=k, m=m, key=key: b.choice(key, float(m), None, f"TX multiplier stage {k}: x{m} (decision 1B: 3 x 2 x 2)"))))
    product = 1.0
    for _, t in mults:
        product *= float(t.value)
    if product != float(n.value):
        raise TemplateRefusal(f"the TX stage multipliers {[float(t.value) for _, t in mults]} multiply to {product:g}, not rf.n_mult = {float(n.value):g}")
    f_ref = plan_value(b, "tx.f_ref", lambda: calc(b, "tx.f_ref", lambda: clock_divided(f_c, n, ("rf.f_c", N_MULT_KEY))))
    stages: list[tuple[str, Traced]] = []
    prev = ("tx.f_ref", f_ref)
    for k, (mkey, m) in enumerate(mults, start=1):
        key = f"tx.f{k}"
        prev = (key, plan_value(b, key, lambda prev=prev, m=m, mkey=mkey, key=key: calc(b, key, lambda: radio.mult_stage(prev[1], m, (prev[0], mkey)))))
        stages.append(prev)
    last = float(stages[-1][1].value)
    if abs(last - float(f_c.value)) > 1e-9 * float(f_c.value):
        raise TemplateRefusal(f"the TX chain's last stage lands on {last:.12g} Hz, not on the carrier {float(f_c.value):.12g} Hz")
    orders: dict[str, tuple[str, Traced]] = {}
    for name, value, text in (("m1", -1.0, "order -1 (the lower neighbour: f - f_T)"), ("p1", 1.0, "order +1 (the upper neighbour: f + f_T)"),
                              ("m2", -2.0, "order -2 (f - 2 f_T)"), ("p2", 2.0, "order +2 (f + 2 f_T)"),
                              ("m3", -3.0, "order -3 (f - 3 f_T)"), ("p3", 3.0, "order +3 (f + 3 f_T)")):
        key = f"tx.order.{name}"
        orders[name] = (key, plan_value(b, key, lambda key=key, value=value, text=text: b.choice(key, value, None, text)))
    z_mid = ("tx.z_mid", plan_value(b, "tx.z_mid", lambda: b.choice(
        "tx.z_mid", 50.0, "ohm", ("the impedance at TX_RAW: the driver pad's matched input (a resistive pi pad into the PHA-1's 50 ohm port model, so "
                                  "50 ohm at every frequency under that model) - the x12 stage's output band-pass is designed into it and the pad's fixture "
                                  "is driven from it"))))
    tol = plan_value(b, "tx.net_tol", lambda: b.choice("tx.net_tol", 1.0, "dB",
                                                       "tolerance of every exact-network fixture row of the TX chain (the netlist realises the designed network)"))
    return TxPlan(f_c=("rf.f_c", f_c), z0=("rf.z0", z0), n_mult=(N_MULT_KEY, n), mults=mults, f_ref=("tx.f_ref", f_ref), stages=stages, orders=orders,
                  z_mid=z_mid, net_tol=tol)


def spur(b: BlockBuilder, key: str, f: tuple[str, Traced], x: tuple[str, Traced], k: tuple[str, Traced]) -> tuple[str, Traced]:
    """``f + k x`` (``calc.rf.mult.spur``) as the parameter ``key`` (shared first)."""
    return key, plan_value(b, key, lambda: calc(b, key, lambda: radio.mult_spur(f[1], x[1], k[1], (f[0], x[0], k[0]))))


# --------------------------------------------------------------------------- the modulator and the multiplier chain (each under its can)


@dataclass
class PmTank:
    """One built phase-modulator tank: its members, bindings, ports and the pins it puts on the source / load nets."""

    members: list[str]
    bindings: dict[str, SpiceBinding]
    inductor: str
    src_pins: list[tuple[str, str]]
    load_pins: list[tuple[str, str]]


class TxModBlock(Block):
    """TCXO -> two buffered PM tanks on one bias node -> ``TX_PM_OUT``, under its own shield can by default (module docstring)."""

    id = MOD_ID
    title = "TX modulator: TCXO and two buffered phase-modulator tanks on one varactor bias node (the integrated audio PM_DRIVE: indirect FM)"
    interface_nets = MOD_INTERFACE

    def __init__(self, shield: str | None = "shield_103", test_points: bool = True) -> None:
        self.shield = shield
        self.test_points = test_points

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        nb = NetBook()
        serves_fc = requirement_ids(ctx, "carrier_frequency")
        serves_tol = requirement_ids(ctx, "frequency_tolerance")
        serves_mod = requirement_ids(ctx, "modulation")
        c = lambda key, value, unit, text: b.choice(key, value, unit, text)  # noqa: E731
        try:
            tp = tx_plan(b)
            rail5_key, rail5 = rail_level(b, "TX_5V")
            rail3_key, rail3 = rail_level(b, "TX_3V3")
            npn_text = b.card(npn_card())
            f_key, f_t = tp.f_ref
            tcxo_r = model_value(b, "model.tcxo.r_out")
            buf_rin = model_value(b, "model.buf.r_in")
            buf_rout = model_value(b, "model.buf.r_out")
            cjo, vj, mg = (model_value(b, f"model.varactor.{x}") for x in ("cjo", "vj", "m"))
            var_card = varactor_card(cjo[1], vj[1], mg[1])
            var_text = b.card(var_card)
            q_pm = model_q(b, float(f_t.value))
            common = common_bias(b)
            bias_analysis(b)
            stab_in = ctx.inputs.get("frequency_tolerance")
            if stab_in is not None:
                stab = ("rf.frequency_tolerance", copy_input(b, "rf.frequency_tolerance", stab_in.traced))
            else:
                stab = ("rf.tcxo_stability", plan_value(b, "rf.tcxo_stability", lambda: b.choice(
                    "rf.tcxo_stability", 2.5, "ppm", "TCXO stability specified for procurement (no frequency_tolerance stated; the profile's 2.5 ppm placeholder) "
                    "[UNVERIFIED: Kyocera KT2520K datasheet]")))
            f_err = calc(b, "tx.f_error", lambda: ppm_offset(tp.f_c[1], stab[1], (tp.f_c[0], stab[0])))
            c_tcxo = c("tx.c_tcxo", 100e-9, "F", "TCXO supply decoupling [UNVERIFIED: Kyocera KT2520K application note]")
            # ---- phase modulator
            c_tot = c("pm.c_tot", 30e-12, "F", "total capacitance of each PM tank (C_fixed + trimmer + varactor at V0; decision 1B)")
            q_l = c("pm.q_l", 10.0, None, "loaded Q of each PM tank (decision 1B: 0.629 rad/V per tank at 2 V with 30 pF)")
            v_bias = c("pm.v_bias", 2.0, "V", "varactor reverse bias V0 at the bias node (the divider from TX_3V3; the bias_nom state)")
            v_lo = c("pm.v_lo", 1.44, "V", ("the bias_lo state of the PM fixtures: V0 - 0.56 V, the varactors' peak swing at 300 Hz and full deviation "
                                            "(2.5 kHz / (12 x 300 Hz x 1.257 rad/V) = 0.553 V, rounded up)"))
            v_hi = c("pm.v_hi", 2.56, "V", "the bias_hi state of the PM fixtures: V0 + 0.56 V (the same swing upward)")
            c_trim = c("pm.c_trim", 5e-12, "F", ("each PM tank's trimmer at its mid position, as simulated (a TZB4 trimmer of about 2-10 pF; the lab trims "
                                                 "the tank's centre offset with it) [UNVERIFIED: Murata TZB4 datasheet]"))
            c_dc = c("pm.c_dc", 1e-9, "F", "DC block in front of each PM tank's series resistor (27 ohm at f_T)")
            c_byp = c("pm.c_bypass", 10e-9, "F", ("RF bypass of the varactor bias node VAR_B (the tanks' inductors return there; 0.43 ohm at f_T - a smaller "
                                                  "bypass detunes the tanks: 1 nF gives about +17 deg, kr447 design §2.4)"))
            r_feed = c("pm.r_feed", 1e3, "ohm", ("bias feed from the divider node to VAR_B: with the bypass a low-pass at 15.9 kHz, outside the voice band "
                                                 "(critic2: a 10 kohm feed puts it at 1.59 kHz and the deviation would not be flat)"))
            c_buf = c("pm.c_buf", 1e-9, "F", "coupling capacitor from each tank to its emitter follower (4.3 ohm at f_T against the follower's 10 kohm)")
            r_div_b = c("pm.r_div_bottom", 3.3e3, "ohm", "lower resistor of the varactor bias divider (TX_3V3 -> PM_BIAS -> GND); the upper one is solved for V0")
            c_audio = c("pm.c_audio", 10e-6, "F", "coupling capacitor PM_DRIVE -> PM_BIAS (the integrated audio onto the bias; a high-pass far below 300 Hz)")
            phase_tol = c("pm.phase_tol", 1.0, "deg", "tolerance of every PM fixture phase row (the exact network phase is the nominal)")
            l_pm = calc(b, "pm.l", lambda: lc_l_for_resonance(f_t, c_tot, (f_key, "pm.c_tot")))
            c_var = {}
            for sid, vkey, tag in PM_STATES:
                v = {"pm.v_lo": v_lo, "pm.v_bias": v_bias, "pm.v_hi": v_hi}[vkey]
                c_var[tag] = calc(b, f"pm.c_var.{tag}", lambda v=v, vkey=vkey: radio.varactor_c_at_bias(
                    cjo[1], vj[1], mg[1], v, (cjo[0], vj[0], mg[0], vkey)))
            c_fix = calc(b, "pm.c_fixed", lambda: radio.pm_c_fixed(c_tot, c_var["nom"], c_trim, ("pm.c_tot", "pm.c_var.nom", "pm.c_trim")))
            dcdv = calc(b, "pm.dc_dv", lambda: radio.varactor_dc_dv(cjo[1], vj[1], mg[1], v_bias, (cjo[0], vj[0], mg[0], "pm.v_bias")))
            k1 = calc(b, "pm.k_pm", lambda: radio.pm_k_pm(q_l, dcdv, c_tot, ("pm.q_l", "pm.dc_dv", "pm.c_tot")))
            k_total = float(k1.value) * 2.0
            plan_value(b, K_PM_KEY, lambda: b.choice(K_PM_KEY, k_total, "rad/V", (
                f"the phase modulator's design constant for the integrator: the two identical tanks on one bias node add, 2 x pm.k_pm = {k_total:.6g} rad/V "
                "(calc.rf.pm.k_pm per tank; no registered calculator adds two slopes, so the sum is shown as this choice) - rf.deviation multiplies the "
                "pm_mod1 / pm_mod2 fixtures' measured chord slopes instead")))
            r_s = {1: calc(b, "pm.r_s1", lambda: radio.pm_source_r_loaded(f_t, l_pm, q_l, q_pm[1], tcxo_r[1], buf_rin[1],
                                                                          (f_key, "pm.l", "pm.q_l", q_pm[0], tcxo_r[0], buf_rin[0]))),
                   2: calc(b, "pm.r_s2", lambda: radio.pm_source_r_loaded(f_t, l_pm, q_l, q_pm[1], buf_rout[1], buf_rin[1],
                                                                          (f_key, "pm.l", "pm.q_l", q_pm[0], buf_rout[0], buf_rin[0])))}
            r_div_t = calc(b, "pm.r_div_top", lambda: divider_r1_for_v_out(rail3, v_bias, r_div_b, (rail3_key, "pm.v_bias", "pm.r_div_bottom")))
            phases: dict[tuple[int, str], Traced] = {}
            for tank, port in ((1, tcxo_r), (2, buf_rout)):
                for sid, vkey, tag in PM_STATES:
                    key = f"pm_mod{tank}.phase.{tag}"
                    phases[(tank, tag)] = calc(b, key, lambda tank=tank, port=port, tag=tag: radio.pm_tank_phase_loaded(
                        f_t, f_t, l_pm, q_pm[1], c_fix, c_trim, c_var[tag], port[1], c_dc, r_s[tank], c_byp, r_feed, buf_rin[1],
                        (f_key, f_key, "pm.l", q_pm[0], "pm.c_fixed", "pm.c_trim", f"pm.c_var.{tag}", port[0], "pm.c_dc", f"pm.r_s{tank}", "pm.c_bypass",
                         "pm.r_feed", buf_rin[0])))
            # ---- audio path PM_DRIVE -> VAR_B (the factor a of rf.deviation)
            f_lo_a = c("pm.couple.f_low", 300.0, "Hz", "pm_couple_300: the audio factor is checked at the low voice-band edge")
            f_ref_a = c("pm.couple.f_ref", 1000.0, "Hz", "pm_couple_1k: the reference tone")
            f_hi_a = c("pm.couple.f_high", 3000.0, "Hz", "pm_couple_3k: the top of the voice band")
            tol_ref = c("pm.couple.tol_ref", 0.3, "dB", "tolerance of pm_couple_1k against its calculator nominal")
            tol_band = c("pm.couple.tol_band", 0.5, "dB", "tolerance of pm_couple_300 / _3k against the 1 kHz nominal (the factor must be flat over the voice band)")
            ac_var = c("pm.ac_variation", "lin", None, "every pm_couple analysis is a single point 'ac lin 1 f f', so every dB row reads an exact sample")
            ac_pts = c("pm.ac_points", 1, None, "one point per pm_couple analysis (at the row's own frequency)")
            r_th = calc(b, "pm.r_th", lambda: parallel_resistance(r_div_t, r_div_b, ("pm.r_div_top", "pm.r_div_bottom")))
            tau_hp = calc(b, "pm.tau_hp", lambda: rc_time_constant(r_th, c_audio, ("pm.r_th", "pm.c_audio")))
            f_hp = calc(b, "pm.f_hp", lambda: radio.emphasis_corner(tau_hp, ("pm.tau_hp",)))
            tau_lp = calc(b, "pm.tau_lp", lambda: rc_time_constant(r_feed, c_byp, ("pm.r_feed", "pm.c_bypass")))
            f_lp = calc(b, "pm.f_lp", lambda: radio.emphasis_corner(tau_lp, ("pm.tau_lp",)))
            hp_ref = calc(b, "pm.couple.hp_ref", lambda: radio.highpass1_db_at(f_ref_a, f_hp, ("pm.couple.f_ref", "pm.f_hp")))
            lp_ref = calc(b, "pm.couple.lp_ref", lambda: radio.lowpass1_db_at(f_ref_a, f_lp, ("pm.couple.f_ref", "pm.f_lp")))
            a_ref = calc(b, "pm.couple.a_ref", lambda: radio.db_sum(hp_ref, lp_ref, ("pm.couple.hp_ref", "pm.couple.lp_ref")))
            lin = plan_value(b, "tx.sweep.lin", lambda: b.choice("tx.sweep.lin", "lin", None, "linear ac sweep variation of the TX fixtures"))
            pts = plan_value(b, "tx.sweep.points", lambda: b.choice("tx.sweep.points", 401, None,
                                                                   "points of each TX fixture sweep (rows are read at their own point analyses, the sweep is the figure)"))
        except ValueError as e:
            raise TemplateRefusal(f"block {self.id}: {e}") from e

        # ---- TCXO
        y = b.part("tcxo", "Y1", part_value(f_t.value),
                   f"TX reference TCXO: {float(f_t.value) / 1e6:.9g} MHz = f_c / N (a custom frequency inside the KT2520K-T range; availability and "
                   f"stability {float(stab[1].value):g} ppm [UNVERIFIED: Kyocera KT2520K datasheet])", [*serves_fc, *serves_tol])
        exclude(b, "Y1", "KT2520K-T TCXO has no SPICE model: excluded (its output is the pm_mod1 fixture's drive port, model.tcxo.r_out)")
        b.leave_open("Y1", "NC", "KT2520K-T pins 2 / 5 are not connected (library type no_connect)")
        c1, _ = passive(b, "cap_0402", "C1", c_tcxo, "TCXO supply decoupling")
        nb.add("TX_3V3", NetKind.POWER, [*y.at("VCC"), ("C1", c1.pin("1"))], "TX_3V3 rail (TCXO supply, varactor bias divider)")
        nb.add("TCXO_OUT", NetKind.ANALOG, y.at("OUT"), "TCXO output: the first PM tank's source")
        nb.add(GROUND_NET, NetKind.GROUND, [*y.at("GND"), ("C1", c1.pin("2"))])
        # ---- shared bias: divider, feed, bypass, audio coupling
        r2, _ = passive(b, "res_0402", "R2", r_div_t, "varactor bias divider (upper, from TX_3V3)")
        r3, _ = passive(b, "res_0402", "R3", r_div_b, "varactor bias divider (lower, to GND)")
        r4, r4_bind = passive(b, "res_0402", "R4", r_feed, "varactor bias feed PM_BIAS -> VAR_B")
        c5, c5_bind = passive(b, "cap_0402", "C5", c_byp, "RF bypass of the varactor bias node VAR_B")
        c6, _ = passive(b, "cap_0603", "C6", c_audio, "audio coupling PM_DRIVE -> PM_BIAS (the integrated audio onto the varactor bias)", serves=serves_mod)
        nb.add("TX_3V3", NetKind.POWER, [("R2", r2.pin("1"))])
        nb.add("PM_BIAS", NetKind.ANALOG, [("R2", r2.pin("2")), ("R3", r3.pin("1")), ("R4", r4.pin("1")), ("C6", c6.pin("2"))],
               "varactor bias divider node (V0 from TX_3V3) with the audio coupled in")
        nb.add("PM_DRIVE", NetKind.ANALOG, [("C6", c6.pin("1"))], "the integrated TX audio from the TX audio block")
        nb.add("VAR_B", NetKind.ANALOG, [("R4", r4.pin("2")), ("C5", c5.pin("1"))], "the bypassed varactor bias node: both PM tanks' inductors return here")
        nb.add(GROUND_NET, NetKind.GROUND, [("R3", r3.pin("2")), ("C5", c5.pin("2"))])
        # ---- PM tanks and followers
        tanks: dict[int, PmTank] = {}
        tank_refs = {1: {"c_dc": "C2", "r_s": "R1", "l": "L1", "c_fix": "C3", "trim": "C_T1", "var": "D1", "c_buf": "C4"},
                     2: {"c_dc": "C8", "r_s": "R9", "l": "L2", "c_fix": "C9", "trim": "C_T2", "var": "D2", "c_buf": "C10"}}
        for tank in (1, 2):
            tanks[tank] = self._pm_tank(b, nb, tank, tank_refs[tank], c_dc=c_dc, r_s=r_s[tank], l_pm=l_pm, c_fix=c_fix, c_trim=c_trim, c_buf=c_buf,
                                        var_card=var_card, var_text=var_text, serves=serves_mod)
        follower_refs = {1: {"r_b1": "R5", "r_b2": "R6", "r_e": "R7", "link": "R8", "c_dec": "C7"},
                         2: {"r_b1": "R10", "r_b2": "R11", "r_e": "R12", "link": "R13", "c_dec": "C11"}}
        for k in (1, 2):
            what = f"PM buffer {k} (MMBT3904 emitter follower after PM tank {k})"
            vals = bias_values(b, f"buf{k}", what, r_b1=15e3, r_b2=33e3, r_e=1.5e3, c_e=None, l_choke=None, choke_note="", common=common,
                               rail_key=rail5_key, rail=rail5)
            st = transistor_stage(b, vals, part="npn_small", value="MMBT3904", q_ref=f"Q{k}", refs=follower_refs[k],
                                  nets={"emitter": f"BUF{k}_E" if k == 1 else PM_OUT_NET, "feed": f"BUF{k}_VC"}, what=what, exp_id=f"tx_ic_buf{k}",
                                  npn_text=npn_text, netbook=nb, serves=serves_mod)
            nb.add(f"BUF{k}_B", NetKind.ANALOG, [*st.base_pins, *tanks[k].load_pins], f"buffer {k} base: the tank's coupling capacitor and the divider")
        nb.add("TCXO_OUT", NetKind.ANALOG, tanks[1].src_pins)
        nb.add("BUF1_E", NetKind.ANALOG, tanks[2].src_pins, "buffer 1 emitter: drives PM tank 2")
        # ---- the PM fixtures
        pm_f_lo = b.choice("pm.sweep.fstart", round(float(f_t.value) * 0.8 / 1e5) * 1e5, "Hz", "start of the PM fixtures' sweep (about 0.8 f_T)")
        pm_f_hi = b.choice("pm.sweep.fstop", round(float(f_t.value) * 1.2 / 1e5) * 1e5, "Hz", "stop of the PM fixtures' sweep (about 1.2 f_T)")
        for tank in (1, 2):
            t = tanks[tank]
            src_net = "TCXO_OUT" if tank == 1 else "BUF1_E"
            src_r = tcxo_r if tank == 1 else buf_rout
            members = [*t.members, "R4", "C5"]
            bindings = {**t.bindings, "R4": r4_bind, "C5": c5_bind}
            states = [RFState(id=sid, port_dc_v={"bias": {"pm.v_lo": v_lo, "pm.v_bias": v_bias, "pm.v_hi": v_hi}[vkey]}) for sid, vkey, _ in PM_STATES]
            rows = [row(f"phase_{tag}", "phase21_deg", "src", "tank", f_t, phases[(tank, tag)], tol=phase_tol, state=sid) for sid, _, tag in PM_STATES]
            sweep = ac_sweep(b, f"pm_mod{tank}_sweep", lin, pts, pm_f_lo, pm_f_hi, f"pm_mod{tank}: 0.8 .. 1.2 f_T")
            b.result.networks.append(RFNetwork(
                id=f"pm_mod{tank}", block=self.id, members=members, bindings=bindings, loss_q={t.inductor: q_pm[1]}, q_ref_hz=f_t,
                ports=[fixture_port("src", src_net, src_r[1], "in"), RFPort(name="tank", net=f"PM{tank}_T", kind="probe", direction="out"),
                       fixture_port("buf", f"BUF{tank}_B", buf_rin[1], "out"), RFPort(name="bias", net="PM_BIAS", kind="control", direction="in")],
                states=states, sweep=[sweep], expectations=rows,
            ))
        # ---- design deck: the audio factor rows
        for aid, f, what in (("pm_ac_low", f_lo_a, "the low voice-band edge"), ("pm_ac_ref", f_ref_a, "the reference tone"), ("pm_ac_high", f_hi_a, "the top of the voice band")):
            b.result.analyses.append(AnalysisSpec(id=aid, kind=SpiceAnalysis.AC, params={"variation": ac_var, "points": ac_pts, "fstart": f, "fstop": f},
                                                  provenance=ctx.provenance(f"single-point ac analysis at {what} (the PM_DRIVE -> varactor factor)")))
        for eid, aid, f, tol in ((PM_COUPLE_IDS[0], "pm_ac_low", f_lo_a, tol_band), (PM_COUPLE_IDS[1], "pm_ac_ref", f_ref_a, tol_ref),
                                 (PM_COUPLE_IDS[2], "pm_ac_high", f_hi_a, tol_band)):
            b.result.expectations.append(Expectation(
                id=eid, analysis_id=aid, vector="v(VAR_B)", reference_vector="v(PM_DRIVE)", reduce=Reduce.DB_AT, at=f, nominal=a_ref, tol_abs=tol,
                provenance=ctx.provenance(f"the audio factor a = |v(VAR_B) / v(PM_DRIVE)| at {float(f.value):g} Hz against its 1 kHz nominal "
                                          "(the C_audio high-pass with the divider's Thevenin R times the feed / bypass low-pass)"),
            ))
        # ---- shield can, test point
        chain = ["Y1", "C_T1", "D1", "Q1", "C_T2", "D2", "Q2"]
        if self.shield is not None:
            sh = b.part(self.shield, "SH2", SHIELD_VALUE.get(self.shield, "RFShield"), "shield can over the phase modulator (TCXO, PM tanks, followers)")
            exclude(b, "SH2", "shield can: no electrical model (its fence pads are GND)")
            nb.add(GROUND_NET, NetKind.GROUND, sh.at("Shield"))
            b.result.shield_ref = "SH2"
        if self.test_points:
            tpp = b.part("testpoint", "TP1", "TestPoint", "alignment test point: the modulated reference (f_T) after the second follower")
            exclude(b, "TP1", "test point: no electrical model")
            nb.add(PM_OUT_NET, NetKind.ANALOG, tpp.at("1"))
        nb.declare(b)
        b.result.ports = [
            RFPort(name=PM_OUT_NET, net=PM_OUT_NET, kind="port", z0_ohm=buf_rout[1], frequency_hz=f_t, direction="out"),
            RFPort(name="TX_5V", net="TX_5V", kind="rail", voltage_v=rail5, direction="in"),
            RFPort(name="TX_3V3", net="TX_3V3", kind="rail", voltage_v=rail3, direction="in"),
        ]
        b.result.chain = chain
        b.result.net_classes = {k: list(v) for k, v in MOD_NET_CLASSES.items()}
        b.result.lab_items += [
            LabItem(id="tx_frequency", block=self.id,
                    what=f"the carrier frequency and its drift over temperature and battery (the TCXO's {float(stab[1].value):g} ppm is {float(f_err.value):.4g} Hz "
                         "at the carrier)", instruments=["frequency counter", "temperature chamber"],
                    reason="the TCXO's stability is a datasheet fact of a custom part [UNVERIFIED]"),
            LabItem(id="tx_procurement_tcxo", block=self.id,
                    what=f"availability of the custom {float(f_t.value) / 1e6:.9g} MHz TCXO (do not order before the channel is confirmed from the official text)",
                    instruments=[], reason="custom frequency; the channel itself is an unverified placeholder of the KR 447 MHz profile"),
            LabItem(id="tx_pm_linearity", block=self.id,
                    what=("the phase modulator's real K_pm and linearity (the two tanks trimmed to centre): audio distortion at 300 Hz / 1 kHz / 3 kHz at "
                          "full deviation, the varactor's real C(V)"),
                    instruments=["modulation analyser", "audio analyser", "C-V meter or VNA"],
                    reason="the varactor is model.varactor (UNVERIFIED); the fixtures judge the tank network, not the part"),
            LabItem(id="tx_deviation", block=self.id, what="peak deviation and modulation response of the transmitter",
                    instruments=["modulation analyser"], reason="rf.deviation multiplies network and principle verdicts under confirmed model values"),
            LabItem(id="tx_obw", block=self.id, what="occupied bandwidth with the prescribed test modulation, adjacent-channel power",
                    instruments=["spectrum analyser", "audio generator"], reason="calc.rf.fm.obw99 is single-tone arithmetic; the KR test method is UNVERIFIED"),
        ]
        return b.done()

    @staticmethod
    def _pm_tank(b: BlockBuilder, nb: NetBook, tank: int, refs: dict[str, str], *, c_dc: Traced, r_s: Traced, l_pm: Traced, c_fix: Traced, c_trim: Traced,
                 c_buf: Traced, var_card: ModelCard, var_text: Traced, serves: list[str]) -> PmTank:
        """One PM tank: DC block -> R_s -> tank node (C_fixed, trimmer, varactor to GND; L to VAR_B) -> coupling to the follower."""
        net = f"pm_mod{tank}"
        what = f"PM tank {tank}"
        members: list[str] = []
        bindings: dict[str, SpiceBinding] = {}
        parts: dict[str, PlacedPart] = {}
        for key, part, value, desc in (("c_dc", "cap_0402", c_dc, "DC block in front of the series resistor"), ("r_s", "res_0402", r_s, "series source resistor (sets Q_L)"),
                                       ("l", "ind_0603", l_pm, "tank inductor (returned to VAR_B)"), ("c_fix", "cap_0402", c_fix, "fixed tank capacitor"),
                                       ("trim", "ctrim", c_trim, "tank trimmer (centre alignment)"), ("c_buf", "cap_0402", c_buf, "coupling to the emitter follower")):
            ref = refs[key]
            parts[ref], bindings[ref] = passive(b, part, ref, value, f"{what}: {desc}", deck=False, network=net, serves=serves if key == "l" else ())
            members.append(ref)
        vref = refs["var"]
        d = b.part("varactor", vref, "D_Capacitance", f"{what}: varactor (reverse biased from VAR_B through the tank inductor)", serves)
        bindings[vref] = card_binding(d, var_card, var_text, b.ctx.provenance(f"{vref}: model.varactor (DVAR, not a vendor model)"))
        exclude(b, vref, FIXTURE_ONLY.format(network=net))
        members.append(vref)
        p = lambda ref, fn: (ref, parts[ref].pin(fn))  # noqa: E731
        node = f"PM{tank}_T"
        nb.add(f"PM{tank}_A", NetKind.ANALOG, [p(refs["c_dc"], "2"), p(refs["r_s"], "1")], f"{what}: between the DC block and the series resistor")
        nb.add(node, NetKind.ANALOG, [p(refs["r_s"], "2"), p(refs["l"], "1"), p(refs["c_fix"], "1"), p(refs["trim"], "1"), (vref, d.pin("K")), p(refs["c_buf"], "1")],
               f"{what}: the tank node (a high-impedance lumped node)")
        nb.add("VAR_B", NetKind.ANALOG, [p(refs["l"], "2")])
        nb.add(GROUND_NET, NetKind.GROUND, [p(refs["c_fix"], "2"), p(refs["trim"], "2"), (vref, d.pin("A"))])
        return PmTank(members=members, bindings=bindings, inductor=refs["l"], src_pins=[p(refs["c_dc"], "1")], load_pins=[p(refs["c_buf"], "2")])


class TxChainBlock(Block):
    """``TX_PM_OUT`` -> x3 -> x2 -> x2 BFR92 stages -> ``TX_RAW``, under the BMI-S-105 can (module docstring).

    Each stage's collector feeds its network through the collector choke,
    which - with the 0 ohm link to ``TX_5V`` - is a member of that network's
    fixture, as is the next stage's base divider: the x3 and x6 stages drive
    double-tuned tanks ``tx_tank1`` / ``tx_tank2``, the x12 stage the
    5-resonator output band-pass ``tx_bpf`` into ``TX_RAW`` (the last tank and
    the band-pass designed as one network: there is no resistive node between
    two filters any more).
    """

    id = CHAIN_ID
    title = "TX multiplier chain: x3 / x2 / x2 BFR92 stages, two double-tuned tanks and the output band-pass (under the shield can)"
    interface_nets = CHAIN_INTERFACE

    def __init__(self, shield: str | None = "shield_105", test_points: bool = True) -> None:
        self.shield = shield
        self.test_points = test_points

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        nb = NetBook()
        serves_fc = requirement_ids(ctx, "carrier_frequency")
        c = lambda key, value, unit, text: b.choice(key, value, unit, text)  # noqa: E731
        names = ("x3", "x6", "x12")
        try:
            tp = tx_plan(b)
            rail5_key, rail5 = rail_level(b, "TX_5V")
            npn_text = b.card(npn_card())
            r_out = model_value(b, "model.bfr92.r_out")
            r_in = model_value(b, "model.bfr92.r_in")
            buf_rout = model_value(b, "model.buf.r_out")
            common = common_bias(b)
            bias_analysis(b)
            qe = ("tx.tank_qe", plan_value(b, "tx.tank_qe", lambda: b.choice(
                "tx.tank_qe", 20.0, None, "end external Q of every double-tuned multiplier tank (decision 1B: about 4.2 dB network loss at Q_u 40; Q_e 25 buys "
                "about 3 dB more rejection per stage for 1 dB more loss)")))
            n_t = ("tx.tank.n", plan_value(b, "tx.tank.n", lambda: b.choice("tx.tank.n", 2.0, None, "resonators per multiplier tank (double-tuned)")))
            lin = plan_value(b, "tx.sweep.lin", lambda: b.choice("tx.sweep.lin", "lin", None, "linear ac sweep variation of the TX fixtures"))
            pts = plan_value(b, "tx.sweep.points", lambda: b.choice("tx.sweep.points", 401, None,
                                                                   "points of each TX fixture sweep (rows are read at their own point analyses, the sweep is the figure)"))
            c_in = c("tx.c_x3_in", 1e-9, "F", "coupling capacitor from the modulator's output TX_PM_OUT to the x3 stage's base (4.3 ohm at f_T)")
            # the three stages' bias first: a network's fixture holds the next stage's base divider, so its values are needed before the network
            vals_by: list[tuple[str, BiasValues]] = []
            for k in range(3):
                m = int(float(tp.mults[k][1].value))
                f_k = tp.stages[k][1]
                what = f"TX multiplier stage {k + 1} (BFR92, x{m} to {float(f_k.value) / 1e6:.6g} MHz)"
                vals_by.append((what, bias_values(b, names[k], what, r_b1=4700.0, r_b2=2200.0, r_e=330.0, c_e=1e-9, l_choke=STAGE_CHOKES[k],
                                                  choke_note=(f"its reactance at {float(f_k.value) / 1e6:.4g} MHz is of the order of the 1 kohm collector port, so it is "
                                                              f"part of the network: a member of the stage's fixture, absorbed by the network's input tap"),
                                                  common=common, rail_key=rail5_key, rail=rail5)))
            # the x12 stage's output band-pass (the last tank and the TX band-pass as one network; kr447 wave-2 review)
            f_key, f_c = tp.f_c
            q_u = model_q(b, float(f_c.value))
            idx = index_choices(b, "tx", 5)
            n_b = ("tx.bpf.n", c("tx.bpf.n", 5.0, None, (
                "resonators of the x12 stage's output band-pass at the carrier: the design's double-tuned last tank and its 3-pole band-pass as one "
                "5-resonator network (two filters joined by their taps, with no resistive node between them, gave 13 dB less rejection at f_c + f_T than "
                "their two fixtures claimed)")))
            bw_b = ("tx.bpf.bw", c("tx.bpf.bw", 25e6, "Hz", ("output band-pass bandwidth (5 resonators, 25 MHz: about the loss and the f_c -/+ f_T rejection "
                                                            "the design's tank + band-pass pair were meant to give together)")))
            l_b = ("tx.bpf.l", c("tx.bpf.l", 27e-9, "H", ("output band-pass resonator inductance (0604HQ): R_p = Q_e w0 L must exceed the collector port's "
                                                        "resistance with its choke - a tap only transforms down")))
            s21_min = c("tx.bpf.s21_min", -13.0, "dB", "tx_bpf s21 at f_c at least -13 dB (the Q-40 network with the collector choke gives about -12.4 dB)")
            margin = c("tx.bpf.margin", 1.0, "dB", "the f_c -/+ f_T rejection rows are bounded at the exact network value less this margin (the design's rule)")
            sub_max = c("tx.bpf.sub_max", -50.0, "dB", "tx_bpf relative level at f_c / 2 (the x2 input residue) at most -50 dB")
            h2_max = c("tx.bpf.h2_max", -20.0, "dB", "tx_bpf relative level at 2 f_c at most -20 dB")
            k2 = ("tx.harm.2", plan_value(b, "tx.harm.2", lambda: c("tx.harm.2", 2.0, None, "harmonic number 2 (the second harmonic of the carrier)")))
            k3h = ("tx.harm.3", plan_value(b, "tx.harm.3", lambda: c("tx.harm.3", 3.0, None, "harmonic number 3")))
            f_2c = ("tx.f_2c", calc(b, "tx.f_2c", lambda: radio.harmonic(f_c, k2[1], (f_key, k2[0]))))
            f2_key, f_2 = tp.stages[1]
            f_3half = ("tx.f_3half", calc(b, "tx.f_3half", lambda: radio.harmonic(f_2, k3h[1], (f2_key, k3h[0]))))
            rows_f = {name: spur(b, f"tx.bpf.f_{name}", tp.f_c, tp.f_ref, tp.orders[name]) for name in ("m1", "p1", "m2", "p2", "m3", "p3")}
            bpf_lo = c("tx.bpf.sweep.fstart", 150e6, "Hz", "start of the output band-pass sweep")
            bpf_hi = c("tx.bpf.sweep.fstop", 1000e6, "Hz", "stop of the output band-pass sweep")
        except ValueError as e:
            raise TemplateRefusal(f"block {self.id}: {e}") from e
        c12, _ = passive(b, "cap_0402", "C12", c_in, "coupling capacitor TX_PM_OUT -> x3 stage")
        nb.add(PM_OUT_NET, NetKind.ANALOG, [("C12", c12.pin("1"))], "the modulated reference from the modulator's second follower")
        # ---- multiplier stages and their networks
        stage_refs = (
            {"r_b1": "R14", "r_b2": "R15", "r_e": "R16", "c_e": "C13", "link": "R17", "c_dec": "C14", "choke": "L3"},
            {"r_b1": "R18", "r_b2": "R19", "r_e": "R20", "c_e": "C20", "link": "R21", "c_dec": "C21", "choke": "L6"},
            {"r_b1": "R22", "r_b2": "R23", "r_e": "R24", "c_e": "C27", "link": "R25", "c_dec": "C28", "choke": "L9"},
        )
        net_parts = ((["C15", "C16", "C17", "C18", "C19"], ["L4", "L5"]), (["C22", "C23", "C24", "C25", "C26"], ["L7", "L8"]),
                     ([f"C{i}" for i in range(29, 40)], [f"L{i}" for i in range(10, 15)]))
        base_extra = [("C12", c12.pin("2"))]
        chain = ["C12"]
        rail_port = RFPort(name="tx_5v", net="TX_5V", kind="rail", voltage_v=rail5, direction="in")
        try:
            for k in range(3):
                name = names[k]
                fk_key, f_k = tp.stages[k]
                what, vals = vals_by[k]
                q_ref = f"Q{k + 3}"
                refs = stage_refs[k]
                st = transistor_stage(b, vals, part="rf_npn", value="BFR92", q_ref=q_ref, refs=refs,
                                      nets={"emitter": f"TX_{name.upper()}_E", "feed": f"TX_{name.upper()}_VC"}, what=what, exp_id=f"tx_ic_{name}",
                                      npn_text=npn_text, netbook=nb, serves=serves_fc)
                nb.add(f"TX_{name.upper()}_B", NetKind.ANALOG, [*base_extra, *st.base_pins], f"stage {k + 1} base: the previous network's output and the divider")
                q_k = model_q(b, float(f_k.value))
                assert vals.l_choke is not None
                caps, inds = net_parts[k]
                collector = f"TX_{name.upper()}_C"
                if k < 2:
                    nxt_key, nxt = names[k + 1], vals_by[k + 1][1]
                    div = calc(b, f"tx.{nxt_key}.r_div", lambda nxt=nxt: parallel_resistance(nxt.r_b1[1], nxt.r_b2[1], (nxt.r_b1[0], nxt.r_b2[0])))
                    r_eff = (f"tx.tank{k + 1}.r_load_eff", calc(b, f"tx.tank{k + 1}.r_load_eff", lambda div=div, nxt_key=nxt_key: parallel_resistance(
                        r_in[1], div, (r_in[0], f"tx.{nxt_key}.r_div"))))
                    port = PortLoad(l_port=vals.l_choke, q_port=q_k, r_load_eff=r_eff)
                    l_key = f"tx.tank{k + 1}.l"
                    l_k = (l_key, b.choice(l_key, TANK_L[k], "H", f"tx_tank{k + 1}: resonator inductance (R_p = Q_e w0 L must exceed the collector port's "
                                                               "resistance with its choke: a tap only transforms down)"))
                    bw_key = f"tx.tank{k + 1}.bw"
                    bw_k = (bw_key, calc(b, bw_key, lambda f_k=f_k, fk_key=fk_key: radio.top_c_bw_for_qe(n_t[1], f_k, qe[1], (n_t[0], fk_key, qe[0]))))
                    net_id = f"tx_tank{k + 1}"
                    top = build_top_c(b, net_id, what=f"tx_tank{k + 1} (double-tuned tank at {float(f_k.value) / 1e6:.6g} MHz)", n=n_t, f0=(fk_key, f_k),
                                      bw=bw_k, l=l_k, r_source=r_out, r_load=r_in, idx=idx, cap_refs=caps, ind_refs=inds,
                                      nodes=[f"TX_T{k + 1}_R1", f"TX_T{k + 1}_R2"], ind_part="ind_0603" if float(f_k.value) < 300e6 else "ind_0604hq",
                                      netbook=nb, port=port)
                    f_m = spur(b, f"tx.tank{k + 1}.f_m", (fk_key, f_k), tp.f_ref, tp.orders["m1"])
                    f_p = spur(b, f"tx.tank{k + 1}.f_p", (fk_key, f_k), tp.f_ref, tp.orders["p1"])
                    s21 = calc(b, f"{net_id}.s21", lambda top=top, q_k=q_k, f_k=f_k, fk_key=fk_key: top.s21(q_k[1], q_k[0], f_k, fk_key))
                    rel_m = calc(b, f"{net_id}.rel_m", lambda top=top, q_k=q_k, f_m=f_m, f_k=f_k, fk_key=fk_key: top.rel(q_k[1], q_k[0], f_m[1], f_m[0], f_k, fk_key))
                    rel_p = calc(b, f"{net_id}.rel_p", lambda top=top, q_k=q_k, f_p=f_p, f_k=f_k, fk_key=fk_key: top.rel(q_k[1], q_k[0], f_p[1], f_p[0], f_k, fk_key))
                    calc(b, f"{net_id}.loss", lambda f_k=f_k, fk_key=fk_key, bw_k=bw_k, q_k=q_k: radio.bpf_dissipation_loss(n_t[1], f_k, bw_k[1], q_k[1],
                                                                                                                             (n_t[0], fk_key, bw_k[0], q_k[0])))
                    f_lo = b.choice(f"tx.tank{k + 1}.sweep.fstart", round(float(f_k.value) * 0.5 / 1e6) * 1e6, "Hz", f"start of tx_tank{k + 1}'s sweep (about f_{k + 1} / 2)")
                    f_hi = b.choice(f"tx.tank{k + 1}.sweep.fstop", round(float(f_k.value) * 1.5 / 1e6) * 1e6, "Hz", f"stop of tx_tank{k + 1}'s sweep (about 1.5 f_{k + 1})")
                    out_net = f"TX_{names[k + 1].upper()}_B"
                    nxt_refs = stage_refs[k + 1]
                    members = [*top.members, refs["choke"], refs["link"], nxt_refs["r_b1"], nxt_refs["r_b2"]]
                    b.result.networks.append(RFNetwork(
                        id=net_id, block=self.id, members=members, bindings=top.bindings,
                        loss_q={**{r: q_k[1] for r in top.inductors}, refs["choke"]: q_k[1]}, q_ref_hz=f_k,
                        ports=[fixture_port("coll", collector, r_out[1], "in"), fixture_port("next", out_net, r_in[1], "out"), rail_port],
                        sweep=[ac_sweep(b, f"{net_id}_sweep", lin, pts, f_lo, f_hi, f"tx_tank{k + 1}: f_{k + 1} / 2 .. 1.5 f_{k + 1}")],
                        expectations=[row("s21", "s21_db", "coll", "next", f_k, s21, tol=tp.net_tol),
                                      row("rel_m", "rel_s21_db", "coll", "next", f_m[1], rel_m, tol=tp.net_tol, ref_at=f_k),
                                      row("rel_p", "rel_s21_db", "coll", "next", f_p[1], rel_p, tol=tp.net_tol, ref_at=f_k)],
                    ))
                    chain += [q_ref, refs["choke"], *caps[:2], inds[0], caps[2], caps[3], inds[1], caps[4]]
                else:
                    port = PortLoad(l_port=vals.l_choke, q_port=q_u, r_load_eff=tp.z_mid)
                    top = build_top_c(b, "tx_bpf", what="tx_bpf (the x12 stage's 5-resonator output band-pass at f_c)", n=n_b, f0=tp.f_c, bw=bw_b, l=l_b,
                                      r_source=r_out, r_load=tp.z_mid, idx=idx, cap_refs=caps, ind_refs=inds, nodes=[f"TX_B_R{i}" for i in range(1, 6)],
                                      ind_part="ind_0604hq", netbook=nb, port=port)
                    s21 = calc(b, "tx_bpf.s21", lambda top=top: top.s21(q_u[1], q_u[0], f_c, f_key))
                    rel = {nm: calc(b, f"tx_bpf.rel_{nm}", lambda top=top, f=f: top.rel(q_u[1], q_u[0], f[1], f[0], f_c, f_key)) for nm, f in rows_f.items()}
                    rel_sub = calc(b, "tx_bpf.rel_sub2", lambda top=top: top.rel(q_u[1], q_u[0], f_2, f2_key, f_c, f_key))
                    rel_2c = calc(b, "tx_bpf.rel_2fc", lambda top=top: top.rel(q_u[1], q_u[0], f_2c[1], f_2c[0], f_c, f_key))
                    calc(b, "tx_bpf.loss", lambda: radio.bpf_dissipation_loss(n_b[1], f_c, bw_b[1], q_u[1], (n_b[0], f_key, bw_b[0], q_u[0])))
                    bound = {nm: calc(b, f"tx_bpf.bound_{nm}", lambda nm=nm: radio.db_sum(rel[nm], margin, (f"tx_bpf.rel_{nm}", "tx.bpf.margin")))
                             for nm in ("m1", "p1")}
                    b.result.networks.append(RFNetwork(
                        id="tx_bpf", block=self.id, members=[*top.members, refs["choke"], refs["link"]], bindings=top.bindings,
                        loss_q={**{r: q_u[1] for r in top.inductors}, refs["choke"]: q_u[1]}, q_ref_hz=f_c,
                        ports=[fixture_port("coll", collector, r_out[1], "in"), fixture_port("pad", "TX_RAW", tp.z_mid[1], "out"), rail_port],
                        sweep=[ac_sweep(b, "tx_bpf_sweep", lin, pts, bpf_lo, bpf_hi, "TX output band-pass: 150-1000 MHz")],
                        expectations=[row("s21_fc", "s21_db", "coll", "pad", f_c, s21_min, bound="at_least"),
                                      row("rel_m1", "rel_s21_db", "coll", "pad", rows_f["m1"][1], bound["m1"], bound="at_most", ref_at=f_c),
                                      row("rel_p1", "rel_s21_db", "coll", "pad", rows_f["p1"][1], bound["p1"], bound="at_most", ref_at=f_c),
                                      row("rel_sub2", "rel_s21_db", "coll", "pad", f_2, sub_max, bound="at_most", ref_at=f_c),
                                      row("rel_2fc", "rel_s21_db", "coll", "pad", f_2c[1], h2_max, bound="at_most", ref_at=f_c)],
                        probes=[probe(f"rel_{nm}", "coll", "pad", rows_f[nm][1], f_c) for nm in ("m2", "p2", "m3", "p3")]
                        + [probe("rel_3half", "coll", "pad", f_3half[1], f_c)],
                    ))
                    chain += [q_ref, refs["choke"], caps[0], *[x for i in range(5) for x in (inds[i], caps[1 + 2 * i], caps[2 + 2 * i])]]
                nb.add(collector, NetKind.ANALOG, [*st.collector_pins, *top.in_pins], f"stage {k + 1} collector: the choke and the network's input tap")
                nb.add(GROUND_NET, NetKind.GROUND, top.gnd_pins)
                base_extra = top.out_pins
            nb.add("TX_RAW", NetKind.RF, base_extra, "the x12 stage's output band-pass into the driver pad (the pad's 50 ohm)")
        except ValueError as e:
            raise TemplateRefusal(f"block {self.id}: {e}") from e
        # ---- shield can, test points
        if self.shield is not None:
            sh = b.part(self.shield, "SH1", SHIELD_VALUE.get(self.shield, "RFShield"), "shield can over the multiplier chain and its output band-pass")
            exclude(b, "SH1", "shield can: no electrical model (its fence pads are GND)")
            nb.add(GROUND_NET, NetKind.GROUND, sh.at("Shield"))
            b.result.shield_ref = "SH1"
        if self.test_points:
            for ref, net, what in (("TP2", "TX_X6_B", "the x3 stage's output (f_1) after tx_tank1"), ("TP3", "TX_X12_B", "the x6 output (f_2) after tx_tank2")):
                tpp = b.part("testpoint", ref, "TestPoint", f"alignment test point: {what}")
                exclude(b, ref, "test point: no electrical model")
                nb.add(net, NetKind.ANALOG, tpp.at("1"))
        nb.declare(b)
        b.result.ports = [
            RFPort(name=PM_OUT_NET, net=PM_OUT_NET, kind="port", z0_ohm=buf_rout[1], frequency_hz=tp.f_ref[1], direction="in"),
            RFPort(name="TX_RAW", net="TX_RAW", kind="port", z0_ohm=tp.z_mid[1], frequency_hz=tp.stages[-1][1], direction="out"),
            RFPort(name="TX_5V", net="TX_5V", kind="rail", voltage_v=rail5, direction="in"),
        ]
        b.result.chain = chain
        b.result.net_classes = {k: list(v) for k, v in CHAIN_NET_CLASSES.items()}
        # ---- plan rows: the spur lines and residues lie outside the band-pass's passband (their levels are lab item tx_spurious)
        for pid, f, note in (("tx_spur_m1", rows_f["m1"][1], "f_c - f_T: the closest multiplier spur below the carrier"),
                             ("tx_spur_p1", rows_f["p1"][1], "f_c + f_T: the closest multiplier spur above the carrier (the weak side of every top-C network)"),
                             ("tx_residue_half", f_2, "f_c / 2: the x2 stage's input residue"), ("tx_residue_3half", f_3half[1], "3/2 f_c: the x2 stage's third harmonic residue")):
            b.result.plan_lines.append(PlanLine(id=pid, kind="margin", f_hz=f, ref_hz=f_c, min_margin_hz=bw_b[1],
                                                note=f"{note}: outside the output band-pass's passband (at least its bandwidth from the carrier); the band-pass's "
                                                     "rejection there is a fixture row of tx_bpf (the collector-to-pad network as built), the level at the output "
                                                     "lab item tx_spurious"))
        b.result.lab_items += [
            LabItem(id="tx_spurious", block=self.id,
                    what="the multiplier spurs at the output (f_c -/+ f_T, -/+ 2 f_T, -/+ 3 f_T, the residues f_c / 2 and 3/2 f_c) against kr447.spurious_max",
                    instruments=["spectrum analyser"],
                    reason="the tanks' and the output band-pass's rejections are network verdicts; the spurs' generation in class-C BJTs and their PM regrowth "
                           "are not modelled"),
            LabItem(id="tx_tank_alignment", block=self.id,
                    what="alignment of the two double-tuned tanks and the 5-resonator output band-pass (part selection; no trimmers; couplings of about 0.15 pF)",
                    instruments=["spectrum analyser", "VNA"], reason="parts tolerances and board parasitics detune the designed networks"),
        ]
        return b.done()


# --------------------------------------------------------------------------- the driver (outside the can)


def pad(b: BlockBuilder, nb: NetBook, name: str, *, refs: tuple[str, str, str], in_net: str, out_net: str, z_pad: tuple[str, Traced], a_db: tuple[str, Traced],
        z_in: tuple[str, Traced], z_out: tuple[str, Traced], f: tuple[str, Traced], sweep: AnalysisSpec, what: str, tol: Traced, s11_max: Traced,
        block_id: str) -> RFNetwork:
    """A matched pi pad (``calc.rf.attenuator.pi.*`` at ``z_pad``): shunt - series - shunt, its fixture with s21 = -A and s11 rows."""
    r_sh = calc(b, f"{name}.r_shunt", lambda: radio.attenuator_pi_r_shunt(z_pad[1], a_db[1], (z_pad[0], a_db[0])))
    r_se = calc(b, f"{name}.r_series", lambda: radio.attenuator_pi_r_series(z_pad[1], a_db[1], (z_pad[0], a_db[0])))
    r_par = calc(b, f"{name}.r_par", lambda: parallel_resistance(r_sh, z_out[1], (f"{name}.r_shunt", z_out[0])))
    ratio = calc(b, f"{name}.ratio", lambda: voltage_divider_ratio(r_se, r_par, (f"{name}.r_series", f"{name}.r_par")))
    s21 = calc(b, f"{name}.s21", lambda: voltage_ratio_to_db(ratio, (f"{name}.ratio",)))
    a, s, c = refs
    pa_, _ = passive(b, "res_0402", a, r_sh, f"{what}: input shunt resistor")
    ps_, _ = passive(b, "res_0402", s, r_se, f"{what}: series resistor")
    pc_, _ = passive(b, "res_0402", c, r_sh, f"{what}: output shunt resistor")
    nb.add(in_net, NetKind.RF, [(a, pa_.pin("1")), (s, ps_.pin("1"))], f"{what}: input")
    nb.add(out_net, NetKind.RF, [(s, ps_.pin("2")), (c, pc_.pin("1"))], f"{what}: output")
    nb.add(GROUND_NET, NetKind.GROUND, [(a, pa_.pin("2")), (c, pc_.pin("2"))])
    return RFNetwork(id=name, block=block_id, members=[a, s, c], ports=[fixture_port("pad_in", in_net, z_in[1], "in"), fixture_port("pad_out", out_net, z_out[1], "out")],
                     sweep=[sweep], expectations=[row("s21", "s21_db", "pad_in", "pad_out", f[1], s21, tol=tol), row("s11", "s11_db", "pad_in", "pad_in", f[1], s11_max, bound="at_most")])


class TxDriverBlock(Block):
    """``TX_RAW`` -> matched pad ``drv_pad`` -> PHA-1 driver -> matched pad ``pa_pad`` -> ``PA_IN`` (module docstring)."""

    id = DRIVER_ID
    title = "TX driver pad, PHA-1 driver and PA-drive pad"
    interface_nets = DRIVER_INTERFACE

    def build_local(self, ctx: BlockContext) -> BlockResult:
        b = BlockBuilder(ctx, self.id, self.title, self.interface_nets)
        nb = NetBook()
        serves_fc = requirement_ids(ctx, "carrier_frequency")
        c = lambda key, value, unit, text: b.choice(key, value, unit, text)  # noqa: E731
        try:
            tp = tx_plan(b)
            rail5_key, rail5 = rail_level(b, "TX_5V")
            f_key, f_c = tp.f_c
            drv = model_value(b, DRIVER_PORT)
            pa_in = model_value(b, PA_INPUT)
            a_drv = ("tx.drv_pad.a_db", c("tx.drv_pad.a_db", 3.0, "dB", ("driver-input pad attenuation: the output band-pass sees the pad's matched input "
                                                                       "(tx.z_mid) whatever the PHA-1's input does")))
            a_pa = ("tx.pa_pad.a_db", c("tx.pa_pad.a_db", 3.0, "dB", ("PA-drive pad attenuation: sets the PA's input level; re-chosen after the conducted "
                                                                   "measurement - a human design change [UNVERIFIED: PHA-1 output level, MMZ09332B gain]")))
            pad_tol = plan_value(b, "tx.pad.tol", lambda: c("tx.pad.tol", 0.2, "dB", "tolerance of the pads' S21 rows"))
            s11_max = plan_value(b, "tx.pad.s11_max", lambda: c("tx.pad.s11_max", -20.0, "dB", "largest S11 of a matched pad"))
            l_ch = c("tx.drv.l_choke", 470e-9, "H", "PHA-1 bias choke from TX_5V into RF_OUT (about 1.32 kohm at f_c) [UNVERIFIED: PHA-1 application circuit, self-resonance]")
            c_dec = c("tx.drv.c_dec", 1e-9, "F", "TX_5V decoupling at the PHA-1 choke")
            c_blk = c("tx.drv.c_block", 100e-12, "F", "PHA-1 input and output DC blocks (3.6 ohm at f_c)")
            lin = plan_value(b, "tx.sweep.lin", lambda: c("tx.sweep.lin", "lin", None, "linear ac sweep variation of the TX fixtures"))
            pts = plan_value(b, "tx.sweep.points", lambda: c("tx.sweep.points", 401, None,
                                                           "points of each TX fixture sweep (rows are read at their own point analyses, the sweep is the figure)"))
            f_lo = c("tx.pad.sweep.fstart", 150e6, "Hz", "start of the TX pad sweeps")
            f_hi = c("tx.pad.sweep.fstop", 1000e6, "Hz", "stop of the TX pad sweeps")
            dsweep = ac_sweep(b, "drv_pad_sweep", lin, pts, f_lo, f_hi, "driver pad: 150-1000 MHz")
            psweep = ac_sweep(b, "pa_pad_sweep", lin, pts, f_lo, f_hi, "PA pad: 150-1000 MHz")
        except ValueError as e:
            raise TemplateRefusal(f"block {self.id}: {e}") from e
        # ---- pads, driver, DC blocks
        drv_net = pad(b, nb, "drv_pad", refs=("R50", "R51", "R52"), in_net="TX_RAW", out_net="DRV_IN", z_pad=drv, a_db=a_drv, z_in=tp.z_mid, z_out=drv,
                      f=tp.f_c, sweep=dsweep, what="driver-input pad", tol=pad_tol, s11_max=s11_max, block_id=self.id)
        pa_net = pad(b, nb, "pa_pad", refs=("R53", "R54", "R55"), in_net="PA_PAD_IN", out_net="PA_IN", z_pad=pa_in, a_db=a_pa, z_in=drv, z_out=pa_in,
                     f=tp.f_c, sweep=psweep, what="PA-drive pad", tol=pad_tol, s11_max=s11_max, block_id=self.id)
        u = b.part("rf_amp", "U50", "PHA-1", "TX driver (gain block before the PA) [UNVERIFIED: Mini-Circuits PHA-1 datasheet]", serves_fc)
        exclude(b, "U50", "PHA-1 has no SPICE model: excluded; its ports are model.driver.port_r in the fixtures")
        lch, _ = passive(b, "ind_0603", "L53", l_ch, "PHA-1 bias choke")
        cd, _ = passive(b, "cap_0402", "C58", c_dec, "TX_5V decoupling at the PHA-1 choke")
        ci, _ = passive(b, "cap_0402", "C57", c_blk, "PHA-1 input DC block")
        co, _ = passive(b, "cap_0402", "C59", c_blk, "PHA-1 output DC block")
        nb.add("DRV_IN", NetKind.RF, [("C57", ci.pin("1"))])
        nb.add("DRV_RFIN", NetKind.RF, [("C57", ci.pin("2")), *u.at("RF_IN")], "PHA-1 input after its DC block")
        nb.add("DRV_OUT", NetKind.RF, [*u.at("RF_OUT"), ("L53", lch.pin("2")), ("C59", co.pin("1"))], "PHA-1 output with its bias choke")
        nb.add("PA_PAD_IN", NetKind.RF, [("C59", co.pin("2"))])
        nb.add("TX_5V", NetKind.POWER, [("L53", lch.pin("1")), ("C58", cd.pin("1"))], "TX_5V rail (PHA-1 bias)")
        nb.add(GROUND_NET, NetKind.GROUND, [*u.at("GND"), ("C58", cd.pin("2"))])
        nb.declare(b)
        b.result.networks += [drv_net, pa_net]
        b.result.ports = [
            RFPort(name="TX_RAW", net="TX_RAW", kind="port", z0_ohm=tp.z_mid[1], frequency_hz=f_c, direction="in"),
            RFPort(name="PA_IN", net="PA_IN", kind="port", z0_ohm=pa_in[1], frequency_hz=f_c, direction="out"),
            RFPort(name="TX_5V", net="TX_5V", kind="rail", voltage_v=rail5, direction="in"),
        ]
        b.result.chain = ["R50", "R51", "R52", "C57", "U50", "L53", "C59", "R53", "R54", "R55"]
        b.result.net_classes = {k: list(v) for k, v in DRIVER_NET_CLASSES.items()}
        del rail5_key, f_key
        b.result.lab_items.append(LabItem(
            id="tx_drive_level", block=self.id, what=f"the PA's drive level through the {float(a_pa[1].value):g} dB pad (the driver's output and the multipliers' conversion)",
            instruments=["power meter", "spectrum analyser"], reason="the PHA-1's output level and the class-C multipliers' conversion are not modelled [UNVERIFIED]"))
        return b.done()


__all__ = [
    "BIAS_ANALYSIS",
    "BIAS_IDS",
    "CHAIN_ID",
    "CHAIN_INTERFACE",
    "CHAIN_NETWORKS",
    "CHAIN_NET_CLASSES",
    "DRIVER_ID",
    "DRIVER_INTERFACE",
    "DRIVER_NETWORKS",
    "DRIVER_NET_CLASSES",
    "DRIVER_PORT",
    "K_PM_KEY",
    "MOD_ID",
    "MOD_INTERFACE",
    "MOD_NETWORKS",
    "MOD_NET_CLASSES",
    "N_MULT_KEY",
    "N_MULT_TEXT",
    "PA_INPUT",
    "PM_COUPLE_IDS",
    "PM_OUT_NET",
    "PM_STATES",
    "RAIL_KEYS",
    "STAGE_MULTIPLIERS",
    "TANK_L",
    "NetBook",
    "PortLoad",
    "TxChainBlock",
    "TxDriverBlock",
    "TxModBlock",
    "TxPlan",
    "build_top_c",
    "calc",
    "copy_input",
    "exclude",
    "fixture_port",
    "model_q",
    "model_value",
    "pad",
    "part_value",
    "passive",
    "plan_value",
    "rail_level",
    "requirement_ids",
    "row",
    "tx_plan",
]
