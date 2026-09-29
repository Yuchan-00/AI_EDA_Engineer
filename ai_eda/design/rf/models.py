"""Model values and SPICE model cards of the KR 447 MHz family: every modelling number no datasheet or measurement grounds.

Invariant: a number that stands in for a part's behaviour - a crystal's
motional values, a varactor's C(V), a PIN diode's R_on / C_off, the port
resistance of an IC that is excluded from the netlist, an inductor's Q, an
op-amp's gain-bandwidth, a generic transistor or diode card - is a **free
choice** of the template, never a fact. Its key starts with
:data:`MODEL_PREFIX` (``model.``), its description ends with
``[UNVERIFIED: <the document or measurement that would ground it>]`` and says
it is "a model value, not a measured part", it enters the IR only through the
``confirm_design`` table (:func:`model_choice`: an ``assumption`` until
confirmed, then ``user_requirement``) and every key a board uses is listed in
``ir.rf.model_values`` (:func:`model_keys`), so ``rf.model_grounding`` can
name it as NOT_VERIFIED until it is grounded. A verdict that rests on these
values says so in its message: :data:`MODEL_VERDICT`.

Model **cards** (the ``.model`` / ``.subckt`` text a SPICE binding carries)
are choices the same way (:func:`card_choice`); the SPICE compiler accepts a
card only with ``user_requirement`` or ``authoritative`` provenance, so an
unconfirmed card never reaches ngspice. A card whose numbers come from model
values is *spelled* from those values' ``Traced`` (``format_spice_number``,
the netlist's own spelling) and names them in :attr:`ModelCard.spelled_from`;
the confirmation table shows the whole card. Numbers that are not model
values - a crystal's L_m - are calculator outputs (:func:`crystal_lm` through
``calc.rf.lc.l_for_resonance``) before a card spells them.

The cards (kr447 design §2.0 "Model-value convention"):

* ``model.npn`` - ``.model QNPN NPN (TR=200n)``: ngspice's default Gummel-Poon
  NPN plus a storage time (the astable template's card), not a BFR92 /
  MMBT3904 vendor model;
* ``model.diode`` - ``.model DG D``: ngspice's default diode, not the 1N4148
  model;
* ``model.pmos`` - a level-1 ``PMOS (VTO=-1 KP=0.5)`` wrapped in a 3-terminal
  subcircuit ``PMOSG3 d g s`` with the bulk tied to the source (an ``M``
  element needs four nodes and the AO3401A symbol has three pins, and a
  binding may not repeat a pin), not the AO3401A model;
* ``model.varactor`` - ``.model DVAR D (CJO=.. VJ=.. M=..)`` spelled from
  ``model.varactor.cjo`` / ``.vj`` / ``.m``, the SPICE junction law
  C = CJO / (1 + V_R / VJ)^M;
* ``model.opamp`` - the single-pole macro ``OPA1P inp inn out``: a VCVS of
  gain ``model.opamp.a0`` into R_g C_g (the pole at GBW / A0), buffered by a
  unit VCVS - the macro the decided splatter decks measured
  (``kr447/decided/splat4_gbw1meg.cir``). It has no supply pins: the op-amp's
  V+ / V- are listed in the binding's ``ignored_pins`` and its output is not
  limited by the rails (a limiter stage must clip with its own diodes);
* ``model.xtal21`` - the crystal's series R_m - L_m - C_m branch with C_0
  across it (``XTAL21 1 2``), from ``model.xtal21.rm`` / ``.cm`` / ``.c0``
  and the calculator's L_m;
* ``model.pot`` - a potentiometer as two resistors at the wiper position
  ``model.pot.position`` (``POT_<ohm>_<permille> a w b``).

Every card is accepted by :mod:`ai_eda.compilers.spice` as written: element
lines (E / R / C / L / M) and a nested ``.model`` live only inside
``.subckt`` ... ``.ends`` (``_normalise_card``); checked by
``tests/test_rf_design_lib.py``, which also runs each card on ngspice when it
is installed (the op-amp's open-loop gain at 1 Hz / 1 MHz, the crystal's
series resonance at 21.4 MHz, the varactor's capacitance at 2 V, the P-FET
switch, the potentiometer's divider).

Inductor Q is one value per frequency band (:data:`INDUCTOR_Q_BANDS`,
:func:`inductor_q_key`): the design needs a Q at every tank frequency
(35.5 / 37.3, 106.5 / 111.9, 213.1 / 223.8, 426.2 / 447.6 MHz) besides
21.4 MHz and 450 kHz, and a frequency outside every band refuses rather than
borrowing a neighbour's Q.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ai_eda.ir import Provenance, SpiceBinding, SpiceDevice, Traced
from ai_eda.tools.calc.rf import lc_l_for_resonance
from ai_eda.tools.calc.si import format_spice_number

from ai_eda.design.base import Choice, choice_provenance
from ai_eda.design.library_parts import TemplateRefusal
from ai_eda.design.rf.parts import PlacedPart

#: every model value / card key starts with this
MODEL_PREFIX = "model."
#: how a verdict that rests on model values words itself
MODEL_VERDICT = "network verdict under confirmed model values (not a measured part)"
#: the phrase every model-value description carries before its ``[UNVERIFIED: ...]`` tail
MODEL_VALUE_PHRASE = "a model value, not a measured part"


@dataclass(frozen=True)
class ModelValue:
    """One numeric model value: what it stands in for and what would ground it."""

    key: str
    value: float
    unit: str | None
    what: str
    #: the datasheet or measurement that would ground it (the ``[UNVERIFIED: ...]`` tail)
    source: str

    def __post_init__(self) -> None:
        if not self.key.startswith(MODEL_PREFIX):
            raise ValueError(f"model value {self.key!r} must start with {MODEL_PREFIX!r}")
        if isinstance(self.value, bool) or not math.isfinite(float(self.value)):
            raise ValueError(f"model value {self.key!r} must be a finite number")

    @property
    def description(self) -> str:
        return f"{self.what} - {MODEL_VALUE_PHRASE} [UNVERIFIED: {self.source}]"


_Q_SOURCE = "Coilcraft / Murata inductor Q data at the tank frequency"

#: every numeric model value of the kr447 design (§2.0 table, the update pass's ``model.buf.*``, the fixture port models)
MODEL_VALUES: dict[str, ModelValue] = {m.key: m for m in (
    ModelValue("model.xtal21.cm", 16e-15, "F",
               "motional capacitance C_m of each 21.4 MHz ladder crystal: C0 / 250, the typical capacitance ratio of an AT-cut fundamental crystal "
               "(the earlier 6 fF, ratio 667, admits no 7.5 kHz ladder - calc.crystal.ladder's C0 bound); a procurement spec the ordered crystals must meet "
               "(L_m follows from the series resonance, calc.rf.lc.l_for_resonance)",
               "crystal datasheet, or a G3UUR measurement of the ordered crystals"),
    ModelValue("model.xtal21.rm", 25.0, "ohm", "motional resistance R_m of each 21.4 MHz ladder crystal", "crystal datasheet, or a G3UUR measurement of the ordered crystals"),
    ModelValue("model.xtal21.c0", 4e-12, "F", "shunt capacitance C_0 of each 21.4 MHz ladder crystal", "crystal datasheet, or a capacitance measurement of the ordered crystals"),
    ModelValue("model.sa605.port_r", 1500.0, "ohm", "port resistance of the SA605 mixer output, IF amplifier input / output and limiter input (the SA605 is excluded from the netlist)",
               "NXP SA605 datasheet"),
    ModelValue("model.l_q.if2", 50.0, None, "inductor Q in the 450 kHz band (IF2 resonators, quadrature coil)", _Q_SOURCE),
    ModelValue("model.l_q.if1", 30.0, None, "inductor Q in the 1-30 MHz band (21.4 MHz ladder matches, diplexer)", _Q_SOURCE),
    ModelValue("model.l_q.hf", 40.0, None, "inductor Q in the 30-100 MHz band (35.5 / 37.3 MHz PM tanks and first multiplier inputs)", _Q_SOURCE),
    ModelValue("model.l_q.vhf", 40.0, None, "inductor Q in the 100-300 MHz band (106.5 / 111.9 and 213.1 / 223.8 MHz multiplier tanks)", _Q_SOURCE),
    ModelValue("model.l_q.uhf", 40.0, None, "inductor Q in the 300-1000 MHz band (426.2 / 447.6 MHz tanks, BPFs, the harmonic LPF, the T/R switch, matches)", _Q_SOURCE),
    ModelValue("model.bfr92.r_out", 1000.0, "ohm",
               ("the BFR92 class-C multiplier transistor's own output resistance at its collector, as a tank or band-pass fixture's source port sees "
                "it; the collector feed choke (with its link and feed decoupling) is not inside it - they are fixture members beside the port"),
               "NXP BFR92AW datasheet (S-parameters)"),
    ModelValue("model.bfr92.r_in", 500.0, "ohm",
               ("the BFR92 class-C multiplier transistor's own input resistance at its base, as a tank fixture's load port sees it; the stage's base "
                "divider is not inside it - it is a fixture member beside the port"),
               "NXP BFR92AW datasheet (S-parameters)"),
    ModelValue("model.varactor.cjo", 20e-12, "F", "zero-bias junction capacitance CJO of the phase-modulator varactor", "varactor datasheet C(V) curve"),
    ModelValue("model.varactor.vj", 0.7, "V", "junction potential VJ of the phase-modulator varactor", "varactor datasheet C(V) curve"),
    ModelValue("model.varactor.m", 0.5, None, "grading coefficient M of the phase-modulator varactor", "varactor datasheet C(V) curve"),
    ModelValue("model.pin.r_on", 1.0, "ohm", "PIN diode resistance at 10 mA bias (the T/R switch's TX state)", "BAR64-03W class PIN diode datasheet"),
    ModelValue("model.pin.c_off", 0.3e-12, "F", "PIN diode capacitance at zero bias (the T/R switch's RX state)", "BAR64-03W class PIN diode datasheet"),
    ModelValue("model.opamp.a0", 1e5, None, "open-loop DC gain of the generic single-pole op-amp macro", "Microchip MCP6001 datasheet"),
    ModelValue("model.opamp.gbw", 1e6, "Hz", "gain-bandwidth product of the generic single-pole op-amp macro", "Microchip MCP6001 datasheet (1 MHz in the library description)"),
    ModelValue("model.buf.r_in", 10e3, "ohm",
               ("input resistance of the MMBT3904 emitter follower between the PM tanks as a tank sees it through its coupling capacitor (enters the "
                "tanks' loaded Q): the follower's base divider in parallel with the transistor's own input (about beta times its emitter load) - the "
                "divider is inside this port model, so no PM fixture carries it as a member, and the model cannot exceed the divider's parallel "
                "resistance"), "MMBT3904 datasheet / a bias-point measurement"),
    ModelValue("model.buf.r_out", 50.0, "ohm",
               ("output resistance of the MMBT3904 emitter follower between the PM tanks (the second tank's drive port): the emitter's own resistance "
                "(r_e plus the source over beta) in parallel with its emitter resistor - the emitter resistor is inside this port model, so the second "
                "PM fixture carries no member for it, and the model cannot exceed that resistor"), "MMBT3904 datasheet / a bias-point measurement"),
    ModelValue("model.tcxo.r_out", 50.0, "ohm", "output resistance of the KT2520K-T TCXO (the PM fixture's drive port)", "Kyocera KT2520K datasheet"),
    ModelValue("model.fuse.r", 0.05, "ohm", "cold resistance of the pack fuse in the design deck", "fuse datasheet"),
    ModelValue("model.k_pm", 1.257, "rad/V", "phase-modulator slope of the two buffered tanks together (audio_ptt only; on tx_exciter it is the pm_mod fixtures' measured slope)",
               "the tx_exciter pm_mod1 / pm_mod2 fixtures, then a modulation-analyser measurement"),
    ModelValue("model.pot.position", 0.5, None, "wiper position of a potentiometer or trimmer in the design deck (a fraction of the track from END1)", "the lab alignment"),
)}

#: (key, lower Hz inclusive, upper Hz exclusive) of each inductor-Q band
INDUCTOR_Q_BANDS: tuple[tuple[str, float, float], ...] = (
    ("model.l_q.if2", 100e3, 1e6),
    ("model.l_q.if1", 1e6, 30e6),
    ("model.l_q.hf", 30e6, 100e6),
    ("model.l_q.vhf", 100e6, 300e6),
    ("model.l_q.uhf", 300e6, 1000e6),
)


def inductor_q_key(f_hz: float) -> str:
    """The ``model.l_q.*`` key whose band holds ``f_hz``; refuses a frequency outside every band (no Q is borrowed from a neighbour)."""
    if isinstance(f_hz, bool) or not math.isfinite(f_hz):
        raise ValueError(f"inductor Q needs a finite frequency, got {f_hz!r}")
    for key, lo, hi in INDUCTOR_Q_BANDS:
        if lo <= f_hz < hi:
            return key
    bands = ", ".join(f"{k} [{lo:.12g}, {hi:.12g}) Hz" for k, lo, hi in INDUCTOR_Q_BANDS)
    raise ValueError(f"no inductor Q model value covers {f_hz:.12g} Hz (bands: {bands}); a Q there would be a guess")


def model_choice(template_id: str, value: str | ModelValue, confirmed: bool) -> tuple[Choice, Traced]:
    """The confirmation-table row and the parameter of a model value (a key of :data:`MODEL_VALUES`, or a block's own :class:`ModelValue`)."""
    mv = MODEL_VALUES[value] if isinstance(value, str) else value
    if "UNVERIFIED" not in mv.description:  # pragma: no cover - the property always writes it
        raise ValueError(f"{mv.key}: a model value's description must say it is unverified")
    unit = f" {mv.unit}" if mv.unit else ""
    traced = Traced(value=float(mv.value), unit=mv.unit, provenance=choice_provenance(template_id, f"{mv.key} = {mv.value!r}{unit}: {mv.description}", confirmed))
    return Choice(mv.key, mv.description, float(mv.value), mv.unit), traced


# --------------------------------------------------------------------------- cards


@dataclass(frozen=True)
class ModelCard:
    """A ``.model`` or ``.subckt`` card a SPICE binding references by ``name``.

    ``device`` is the element the binding uses (``X`` for a subcircuit);
    ``functions`` the parts-table functions in the element's node order
    (:mod:`ai_eda.design.rf.parts`); ``spelled_from`` the model-value keys
    whose numbers the text spells.
    """

    key: str
    name: str
    text: str
    device: SpiceDevice
    functions: tuple[str, ...]
    what: str
    source: str
    spelled_from: tuple[str, ...] = ()

    @property
    def description(self) -> str:
        text = self.text.replace("\n", " / ")
        spelled = f" (spelled from {', '.join(self.spelled_from)})" if self.spelled_from else ""
        return f"{self.what}: `{text}`{spelled} - a model card, not a vendor model [UNVERIFIED: {self.source}]"


def _n(t: Traced | float, what: str) -> str:
    value = t.value if isinstance(t, Traced) else t
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{what} must be a finite number, got {value!r}")
    return format_spice_number(float(value))


def npn_card() -> ModelCard:
    return ModelCard("model.npn", "QNPN", ".model QNPN NPN (TR=200n)", SpiceDevice.Q, ("C", "B", "E"),
                     "generic Gummel-Poon NPN (ngspice defaults, TR = 200 ns storage time; the astable template's card)", "BFR92 / MMBT3904 vendor models")


def diode_card() -> ModelCard:
    return ModelCard("model.diode", "DG", ".model DG D", SpiceDevice.D, ("A", "K"), "ngspice's default diode", "1N4148WS vendor model")


def pmos_card() -> ModelCard:
    text = "\n".join((".subckt PMOSG3 d g s", "M1 d g s s PMOSG", ".model PMOSG PMOS (VTO=-1 KP=0.5)", ".ends"))
    return ModelCard("model.pmos", "PMOSG3", text, SpiceDevice.X, ("D", "G", "S"),
                     "level-1 PMOS (VTO = -1 V, KP = 0.5 A/V^2) as a 3-terminal subcircuit with the bulk tied to the source", "AOS AO3401A vendor model")


def pa_supply_card(r_on: Traced, r_off: Traced, v_pd: Traced) -> ModelCard:
    """``.subckt PASUP v g pd``: the PA (no RF model) as its supply draw, switched by its POWER_DOWN pin.

    ``R_on`` from VCC1 through a voltage-controlled switch that is closed
    while V(pd) is below ``v_pd`` (the PA powered up) and ``R_off`` across
    VCC1 always (its powered-down draw) - so a supply discharge that the
    circuit's own POWER_DOWN edge switches off is not credited to the PA.
    The switch has 0.1 V of hysteresis around ``v_pd`` (without any,
    ngspice-42's ideal switch stopped a 120 ms PTT transient with "timestep
    too small" at the edge). The node names avoid ``gnd`` (ngspice's alias of
    node 0).
    """
    for t, what in ((r_on, "R_on"), (r_off, "R_off"), (v_pd, "the POWER_DOWN threshold")):
        if not (isinstance(t.value, (int, float)) and not isinstance(t.value, bool) and t.value > 0):
            raise ValueError(f"PA supply model: {what} must be positive, got {t.value!r}")
    if not r_off.value > r_on.value:
        raise ValueError("PA supply model: the powered-down draw R_off must exceed the powered-up R_on")
    text = "\n".join((".subckt PASUP v g pd", f"Ron v a {_n(r_on, 'R_on')}", "S1 a g g pd SWPD", f"Roff v g {_n(r_off, 'R_off')}",
                      f".model SWPD SW (VT=-{_n(v_pd, 'V_pd')} VH=0.1 RON=1m ROFF=1e12)", ".ends"))
    return ModelCard("model.pa.supply", "PASUP", text, SpiceDevice.X, ("VCC1", "GND", "POWER_DOWN"),
                     "the PA as its supply draw from VCC1: R_on (model.pa.r_supply) while POWER_DOWN is below model.pa.v_pd, only R_off "
                     "(model.pa.r_off) while it is above (a voltage-controlled switch, 0.1 V hysteresis; no RF, no bias pins)",
                     "NXP MMZ09332B datasheet (supply currents powered up and down, POWER_DOWN levels)",
                     ("model.pa.r_supply", "model.pa.r_off", "model.pa.v_pd"))


def varactor_card(cjo: Traced, vj: Traced, m: Traced) -> ModelCard:
    """``.model DVAR D (CJO=.. VJ=.. M=..)`` spelled from the three model values (C = CJO / (1 + V_R / VJ)^M)."""
    for t, what in ((cjo, "CJO"), (vj, "VJ")):
        if not (isinstance(t.value, (int, float)) and t.value > 0):
            raise ValueError(f"varactor {what} must be positive, got {t.value!r}")
    if not (isinstance(m.value, (int, float)) and 0 < m.value < 1):
        raise ValueError(f"varactor grading coefficient M must lie in (0, 1), got {m.value!r}")
    text = f".model DVAR D (CJO={_n(cjo, 'CJO')} VJ={_n(vj, 'VJ')} M={_n(m, 'M')})"
    return ModelCard("model.varactor", "DVAR", text, SpiceDevice.D, ("A", "K"), "varactor as a junction diode", "varactor datasheet C(V) curve",
                     ("model.varactor.cjo", "model.varactor.vj", "model.varactor.m"))


def opamp_card(a0: Traced, gbw: Traced, r_g_ohm: float = 1000.0) -> ModelCard:
    """The single-pole macro: E_g = A0 (v+ - v-), pole R_g C_g at GBW / A0, unit-gain output VCVS (no supply pins, no rail limit)."""
    if not (isinstance(a0.value, (int, float)) and a0.value > 1):
        raise ValueError(f"op-amp A0 must exceed 1, got {a0.value!r}")
    if not (isinstance(gbw.value, (int, float)) and gbw.value > 0):
        raise ValueError(f"op-amp GBW must be positive, got {gbw.value!r}")
    c_g = a0.value / (2.0 * math.pi * r_g_ohm * gbw.value)  # the pole f_p = GBW / A0 = 1 / (2 pi R_g C_g)
    text = "\n".join((".subckt OPA1P inp inn out", f"Eg g 0 inp inn {_n(a0, 'A0')}", f"Rg g p {_n(r_g_ohm, 'R_g')}", f"Cg p 0 {_n(c_g, 'C_g')}",
                      "Eo out 0 p 0 1", ".ends"))
    return ModelCard("model.opamp", "OPA1P", text, SpiceDevice.X, ("+", "-", "OUT"),
                     "generic single-pole op-amp macro (gain A0, pole at GBW / A0 = 1 / (2 pi R_g C_g), no supply pins, output not rail-limited)",
                     "Microchip MCP6001 vendor model", ("model.opamp.a0", "model.opamp.gbw"))


def crystal_lm(f_s: Traced, cm: Traced, ids: tuple[str, str] = ("rf.if1", "model.xtal21.cm")) -> Traced[float]:
    """L_m = 1 / ((2 pi f_s)^2 C_m): the motional inductance that puts the series resonance at ``f_s`` (``calc.rf.lc.l_for_resonance``)."""
    return lc_l_for_resonance(f_s, cm, ids)


def crystal_card(lm: Traced, cm: Traced, rm: Traced, c0: Traced, *, key: str = "model.xtal21", name: str = "XTAL21") -> ModelCard:
    """``.subckt <name> 1 2``: R_m - L_m - C_m in series between the pins, C_0 across them (the crystal's Butterworth-Van Dyke model)."""
    for t, what in ((lm, "L_m"), (cm, "C_m"), (rm, "R_m"), (c0, "C_0")):
        if not (isinstance(t.value, (int, float)) and not isinstance(t.value, bool) and t.value > 0):
            raise ValueError(f"crystal {what} must be positive, got {t.value!r}")
    text = "\n".join((f".subckt {name} 1 2", f"Lm 1 a {_n(lm, 'L_m')}", f"Cm a b {_n(cm, 'C_m')}", f"Rm b 2 {_n(rm, 'R_m')}", f"C0 1 2 {_n(c0, 'C_0')}", ".ends"))
    return ModelCard(key, name, text, SpiceDevice.X, ("1", "2"), "crystal as its motional series R-L-C branch with C_0 across it (Butterworth-Van Dyke)",
                     "crystal datasheet, or a G3UUR measurement of the ordered crystals", (f"{key}.cm", f"{key}.rm", f"{key}.c0"))


def potentiometer_card(r_total: Traced | float, position: Traced | float, *, key: str = "model.pot", name: str | None = None) -> ModelCard:
    """Two resistors ``END1 - WIPER - END3`` at the wiper ``position`` (a fraction of the track from END1, strictly between 0 and 1).

    ``key`` is the card's confirmation-table row: one row per distinct card, so
    a board with several pots of different resistance or position gives each
    its own key (``model.pot.vol``, ``model.pot.sql``); the builder refuses a
    second, different card under a key that already holds one.
    """
    r = r_total.value if isinstance(r_total, Traced) else r_total
    pos = position.value if isinstance(position, Traced) else position
    if not (isinstance(r, (int, float)) and r > 0 and math.isfinite(r)):
        raise ValueError(f"potentiometer resistance must be positive, got {r!r}")
    if not (isinstance(pos, (int, float)) and 0 < pos < 1):
        raise ValueError(f"wiper position must lie strictly between 0 and 1 (ngspice simulates a 0 ohm resistor as a different part), got {pos!r}")
    name = name or f"POT_{round(r)}_{round(pos * 1000)}"
    text = "\n".join((f".subckt {name} a w b", f"Ra a w {_n(r * pos, 'R_a')}", f"Rb w b {_n(r * (1 - pos), 'R_b')}", ".ends"))
    return ModelCard(key, name, text, SpiceDevice.X, ("END1", "WIPER", "END3"), "potentiometer as two resistors at the wiper position",
                     "the lab alignment", ("model.pot.position",))


def card_choice(template_id: str, card: ModelCard, confirmed: bool) -> tuple[Choice, Traced[str]]:
    """The confirmation-table row of a card and the ``Traced`` card text a binding carries (``user_requirement`` once confirmed)."""
    return Choice(card.key, card.description), Traced(value=card.text, provenance=choice_provenance(template_id, f"{card.key}: {card.description}", confirmed))


def card_binding(
    placed: PlacedPart,
    card: ModelCard,
    card_text: Traced[str],
    provenance: Provenance,
    *,
    functions: tuple[str, ...] | None = None,
    ignored: dict[str, str] | None = None,
) -> SpiceBinding:
    """The SPICE binding of ``placed`` on ``card``: node order from the parts-table functions, ``ignored`` functions' pins listed with the reason.

    Refuses a card text that is not the card's (the binding would simulate
    another model than the table shows).
    """
    if card_text.value != card.text:
        raise TemplateRefusal(f"{placed.ref}: the card text does not match {card.key} ({card.name})")
    order = [placed.pin(f) for f in (functions or card.functions)]
    ignored_pins = {pin: reason for f, reason in sorted((ignored or {}).items()) for pin in placed.pins(f)}
    return SpiceBinding(device=card.device, model_name=card.name, model_card=card_text, pin_order=order, ignored_pins=ignored_pins, provenance=provenance)


#: the reason the op-amp binding gives for its supply pins
OPAMP_SUPPLY_IGNORED = "supply pin: the model.opamp macro has no supply dependence (its output is not limited by the rails)"
#: the reason the potentiometer binding gives for its mounting pin
POT_MOUNT_IGNORED = "mounting pin: no electrical function in the potentiometer model"


def model_keys(parameter_keys, cards: list[ModelCard] | tuple[ModelCard, ...] = ()) -> list[str]:
    """The sorted ``model.*`` keys among ``parameter_keys`` plus the keys of ``cards``: what ``ir.rf.model_values`` lists."""
    return sorted({k for k in parameter_keys if k.startswith(MODEL_PREFIX)} | {c.key for c in cards})


__all__ = [
    "INDUCTOR_Q_BANDS",
    "MODEL_PREFIX",
    "MODEL_VALUES",
    "MODEL_VALUE_PHRASE",
    "MODEL_VERDICT",
    "OPAMP_SUPPLY_IGNORED",
    "POT_MOUNT_IGNORED",
    "ModelCard",
    "ModelValue",
    "card_binding",
    "card_choice",
    "crystal_card",
    "crystal_lm",
    "diode_card",
    "inductor_q_key",
    "model_choice",
    "model_keys",
    "npn_card",
    "opamp_card",
    "pa_supply_card",
    "pmos_card",
    "potentiometer_card",
    "varactor_card",
]
