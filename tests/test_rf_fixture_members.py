"""Fixture membership on every kr447 build (the fixture-membership pass, part A1).

An RF fixture (``ir.rf.networks``) simulates only its members between its
port models, so every part the board hangs on a network's nets that is not a
member is a part the verdict never saw. The rules and the finders are
``tests/rf_fixture_audit.py``; this module applies them to all six
``radio_build`` values (``audio_ptt``, ``rx_backend``, ``rx_frontend``,
``tx_exciter``, ``transceiver``, ``transceiver_conducted``) and the options
that change what a build composes - the time-out's PTT timer, the
transceiver with a confirmed ``antenna_impedance`` (the only build that holds
the ``ant_match`` network) and with every optional input, the stage boards'
bench stand-ins - every fixture network of every template:

* no part on a network's internal nets and no passive on its port / probe
  nets outside the network, unless another network holds it together with
  the network's parts there (a cascade fixture - ``pa_lpf``, ``ant_end`` -
  or an enclosing one - ``if1_filter`` around ``if1_ladder``), or it is in
  :data:`ALLOWED_OUTSIDERS` with its recorded reason;
* no two networks joined tap to tap: two networks meet on a signal net only
  nested, inside a third network that holds both sides, or at a junction of
  :data:`ALLOWED_JUNCTIONS` with its recorded reason.

Every allowance names its kind, and the test checks the kind's claim on the
IR, so a reason cannot stay true by being written down: a **port model**
(the port's resistance carries a ``model.*`` value whose text says the part
is inside it - the fixture records the statement in the port's provenance -
``rf.model_grounding`` names that text, and the model cannot exceed the
parts it includes); a **bypass** (a node both networks hold a shared
capacitor to GND on, whose impedance at the network's reference frequency is
at most 5 % of the branch it grounds - the outsider inductor and the rest of
the other network behind it, that network's ports at their port models,
solved in every state by a linear nodal analysis of the fixture circuit: an
inductor that runs into the other tank's own capacitance is series-resonant
there, far below its reactance - and with that branch hung on the node
passively every row of the network moves by at most a tenth of its
tolerance; what a *driven* branch couples in is not passive, so a bypass
allowance names the lab item that carries it); a **matched pad** (a
network of resistors only whose s11 row, driven at the shared net, bounds
its input match, with the same port resistance on both sides - its
S-parameters do not depend on what drives it, and the other network's load
model is its input); a **DC block** (a series capacitor into the part the
port model stands for, with at most a bias choke to a rail behind it, whose
combined effect on |S21| is at most 0.05 dB at the network's lowest row
frequency). No entry may go unused. An allowance that rests on a lab item
is used only where the design carries that lab item. A transistor's port
model beside its bias parts (``model.bfr92.*``, ``model.lna.port_r``,
``model.ifamp.port_r``) says they are not inside it and every one of them is
a member; a port model that says a part is inside it has no member for that
part - nothing counts twice.

The two PM tanks' coupling through their shared bias node is checked
against what the texts and the lab item ``tx_pm_coupling`` say: the branch
share, the passive bound and the linear estimate of the driven coupling.

With ngspice the changed fixtures are run again and their measured values
pinned (``fe_bpf2`` / ``fe_bpf3`` / ``diplexer`` on the rx_frontend build,
``pa_lpf`` on the tx_exciter build, ``ant_end`` on the transceiver builds),
and the joint passive PM circuit is run to confirm the nodal solve.
"""

from __future__ import annotations

import cmath
import math
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from ai_eda.agents.base import IRProposal
from ai_eda.agents.requirement import _answer_requirement
from ai_eda.design import templates as templates_mod
from ai_eda.design.board import add_board
from ai_eda.design.inputs import read_inputs
from ai_eda.ir import CircuitIR, NetKind, ProjectMeta
from ai_eda.ir import ValidationStatus as S
from ai_eda.ir.rf import RFNetwork
from ai_eda.tools.calc.part_value import parse_part_value
from ai_eda.tools.calc.radio import varactor_c_f
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.spice import NgspiceShared
from ai_eda.tools.spice.rf_fixture import spice_rf_results
from ai_eda.validation.rf import model_grounding_result
from ai_eda.workflow.orchestrator import Orchestrator
from tests.rf_fixture_audit import GROUND, PASSIVE_REF, Junction, Outsider, junctions, net_refs, outsiders, signal_nets

runner = NgspiceShared()
needs_ngspice = pytest.mark.skipif(not runner.available(), reason="ngspice shared library not found")
_REAL = KicadLibrary()
HAS_LIBS = all(_REAL.symbol_file(lib) is not None for lib in ("RF_Amplifier", "RF_AM_FM", "RF_Mixer", "Oscillator", "Device", "Connector", "Amplifier_Audio")) and \
    _REAL.footprint_file("RF_Shielding", "Laird_Technologies_BMI-S-105_38.10x25.40mm") is not None and \
    _REAL.footprint_file("Connector_Wire", "SolderWire-0.5sqmm_1x01_D0.9mm_OD2.1mm") is not None
needs_libs = pytest.mark.skipif(not HAS_LIBS, reason="KiCad 10 libraries with the RF parts not installed (set KICAD10_SYMBOL_DIR)")

_TX = {"modulation": "fm", "input_voltage": "7.4 V", "carrier_frequency": "447.5625 MHz"}
#: every optional key of the transceiver's family row stated at once (the antenna match, the time-out, the radiated numbers)
_EVERYTHING = {**_TX, "antenna_gain": "2.15 dBi", "link_range": "1 km", "antenna_impedance": "36 ohm", "tx_timeout": "180 s", "occupied_bandwidth": "8.5 kHz",
               "channel_spacing": "12.5 kHz", "rx_sensitivity": "-113 dBm", "frequency_deviation": "2.5 kHz", "audio_bandwidth": "3 kHz", "tx_power": "0.5 W",
               "erp": "0.5 W", "field_strength_limit": "0.01 V/m", "frequency_tolerance": "2.5 ppm", "system_impedance": "50 ohm"}
#: every radio_build with the least inputs it builds from, and the options that change what a build composes: the time-out (the PTT timer),
#: the antenna match (only with a confirmed antenna_impedance), every optional input at once, and the bench stand-ins of the stage boards
BUILDS: dict[str, dict[str, str]] = {
    "audio_ptt": {"radio_build": "audio_ptt", "modulation": "fm", "input_voltage": "7.4 V"},
    "rx_backend": {"radio_build": "rx_backend", "modulation": "fm", "input_voltage": "7.4 V"},
    "rx_frontend": {"radio_build": "rx_frontend", "carrier_frequency": "447.5625 MHz", "input_voltage": "7.4 V"},
    "rx_frontend_bench": {"radio_build": "rx_frontend", "carrier_frequency": "447.5625 MHz", "input_voltage": "7.4 V"},
    "tx_exciter": {"radio_build": "tx_exciter", **_TX},
    "tx_exciter_timeout": {"radio_build": "tx_exciter", **_TX, "tx_timeout": "180 s"},
    "tx_exciter_bench": {"radio_build": "tx_exciter", **_TX},
    "transceiver": {"radio_build": "transceiver", **_TX},
    "transceiver_conducted": {"radio_build": "transceiver_conducted", **_TX},
    "transceiver_antenna_match": {"radio_build": "transceiver", **_TX, "antenna_impedance": "36 ohm"},
    "transceiver_everything": {"radio_build": "transceiver", **_EVERYTHING},
}


def _bench_template(build: str):
    """The stage boards' bench stand-ins (the template tests' compositions); ``None``: the registered template the requirements select."""
    if build == "rx_frontend_bench":
        from ai_eda.design.rf.t_rx_frontend import BenchSupplyBlock, KR447RxFrontendTemplate

        return KR447RxFrontendTemplate(companions=(BenchSupplyBlock(),))
    if build == "tx_exciter_bench":
        from ai_eda.design.rf.t_tx_exciter import Kr447TxExciterTemplate, bench_companions

        return Kr447TxExciterTemplate(companions=bench_companions())
    return None
_RX_FE = ["fe_bpf2", "fe_bpf3", "diplexer", "lo_tank1", "lo_tank2", "lo_bpf", "lo_pad"]
_RX_BE = ["if1_ladder", "if1_filter", "if2_bpf_a", "if2_bpf_b", "quad_tank"]
_TX_NETS = ["pm_mod1", "pm_mod2", "tx_tank1", "tx_tank2", "tx_bpf", "drv_pad", "pa_pad", "pa_match"]
_TRX = sorted({*_RX_FE, *_RX_BE, *_TX_NETS, "trsw", "lpf", "ant_end"})
#: the fixture networks each build holds
NETWORKS: dict[str, list[str]] = {
    "audio_ptt": [],
    "rx_backend": sorted(_RX_BE),
    "rx_frontend": sorted(_RX_FE),
    "rx_frontend_bench": sorted(_RX_FE),
    "tx_exciter": sorted([*_TX_NETS, "lpf", "pa_lpf"]),
    "tx_exciter_timeout": sorted([*_TX_NETS, "lpf", "pa_lpf"]),
    "tx_exciter_bench": sorted([*_TX_NETS, "lpf", "pa_lpf"]),
    "transceiver": _TRX,
    "transceiver_conducted": _TRX,
    "transceiver_antenna_match": sorted([*_TRX, "ant_match"]),
    "transceiver_everything": sorted([*_TRX, "ant_match"]),
}

#: the allowance kinds (module docstring)
PORT_MODEL, BYPASS, PAD, DC_BLOCK = "port model", "bypass", "matched pad", "DC block"
#: what a port model's text says when a divider / an emitter resistor on its net is inside it (``ai_eda.design.rf.models``)
DIVIDER_INSIDE = "the divider is inside this port model"
EMITTER_R_INSIDE = "the emitter resistor is inside this port model"
#: what a transistor port model's text says when the bias parts on its net are fixture members instead (``model.bfr92.*``, ``model.lna.port_r``,
#: ``model.ifamp.port_r``: the transistor's own resistance, so a value measured for the whole stage would count the divider / choke twice)
BIAS_BESIDE = ("not inside it", "fixture member", "beside the port")
#: every model value a port stands on whose text must say its bias parts are fixture members beside it
TRANSISTOR_PORTS = ("model.bfr92.r_in", "model.bfr92.r_out", "model.lna.port_r", "model.ifamp.port_r")
#: a bypass is an RF ground for a branch when its impedance is at most this share of the branch's (the node's impedance then changes by at
#: most about that share when the branch is hung on it)
BYPASS_SHARE = 0.05
#: ... and the network's rows, with the branch hung on the node passively, move by at most this share of their tolerance
JOINT_TOL_SHARE = 0.1
#: a DC block (and a bias choke behind it) may change |S21| by at most this much (dB) at the network's lowest row frequency
DC_BLOCK_DB = 0.05
#: the model value a port's resistance carries: its provenance note reads ``... model.<key> = <value> <unit>: <text>`` (``models.model_choice``)
MODEL_KEY_IN_NOTE = re.compile(r"\b(model\.[A-Za-z0-9_.]+) = ")


@dataclass(frozen=True)
class Allow:
    kind: str
    reason: str
    #: PORT_MODEL: the words the port model's text says
    phrase: str = ""
    #: a lab item the reason rests on (it must be in ir.rf.lab_items wherever the allowance is used)
    lab_item: str = ""


_FOLLOWER_DIVIDER = ("the emitter follower's base divider on the PM fixture's load port: model.buf.r_in is stated to be the follower's input with its "
                     "divider, so the port model stands for it (and cannot exceed the divider's parallel resistance)")
_OTHER_TANK = ("the other PM tank's inductor on the shared varactor bias node VAR_B: both fixtures hold its RF bypass pm.c_bypass, an RF ground for "
               "the passive branch - the inductor runs into the other tank's own capacitance and is series-resonant at f_T (about 14-15 ohm), the "
               "bypass's 0.43 ohm about 3 % of it, and hung on passively the branch moves the fixture's phases by at most 0.032 deg; the driven "
               "coupling (follower 1 feeds tank 2 with tank 1's own signal, about 1 deg on tank 1 and 2.85 deg on tank 2) is the lab item's")
_PHA1_OUT = ("the PHA-1's output DC block (its bias choke to the rail behind it): the pad's source port stands for the PHA-1 output seen through them; "
             "their effect on |S21| is well under 0.05 dB at the pad's frequency")
#: (network, ref) -> the allowance of a part outside a network on one of its nets
ALLOWED_OUTSIDERS: dict[tuple[str, str], Allow] = {
    ("pm_mod1", "R805"): Allow(PORT_MODEL, _FOLLOWER_DIVIDER, DIVIDER_INSIDE),
    ("pm_mod1", "R806"): Allow(PORT_MODEL, _FOLLOWER_DIVIDER, DIVIDER_INSIDE),
    ("pm_mod2", "R810"): Allow(PORT_MODEL, _FOLLOWER_DIVIDER, DIVIDER_INSIDE),
    ("pm_mod2", "R811"): Allow(PORT_MODEL, _FOLLOWER_DIVIDER, DIVIDER_INSIDE),
    ("pm_mod2", "R807"): Allow(PORT_MODEL, ("the first follower's emitter resistor on the second PM fixture's source port: model.buf.r_out is stated to be "
                                            "the follower's output with its emitter resistor (and cannot exceed it)"), EMITTER_R_INSIDE),
    ("pm_mod1", "L802"): Allow(BYPASS, _OTHER_TANK, lab_item="tx_pm_coupling"),
    ("pm_mod2", "L801"): Allow(BYPASS, _OTHER_TANK, lab_item="tx_pm_coupling"),
    ("tx_bpf", "R850"): Allow(PAD, "the driver pad's input resistors: the band-pass's load port stands for the matched pad, whose own s11 row proves its input "
                                   "is the port's resistance"),
    ("tx_bpf", "R851"): Allow(PAD, "the driver pad's series resistor (as R850)"),
    ("drv_pad", "C839"): Allow(PAD, ("the band-pass's output tap on the pad's input: a matched resistive pad's S-parameters do not depend on what drives it, "
                                     "and the band-pass's own fixture is loaded by the pad's input resistance")),
    ("lo_pad", "C756"): Allow(DC_BLOCK, _PHA1_OUT),
    ("pa_pad", "C859"): Allow(DC_BLOCK, _PHA1_OUT),
    ("drv_pad", "C857"): Allow(DC_BLOCK, "the TX driver PHA-1's input DC block: the pad's load port stands for the PHA-1 input seen through it"),
    ("pa_pad", "C907"): Allow(DC_BLOCK, "the PA's input DC block: the pad's load port stands for the PA input (model.pa.r_in) seen through it"),
    ("if1_filter", "C654"): Allow(DC_BLOCK, ("the IF1 post-amp's output DC block (its collector choke to the feed behind it): the filter's source port stands "
                                             "for the post-amp's output, which the IF1 port only declares at the system impedance - no output match is "
                                             "designed, the lab item carries it"), lab_item="ifamp_output"),
}
#: (net, network, network) with the two networks sorted -> the allowance of a junction no third network holds
ALLOWED_JUNCTIONS: dict[tuple[str, str, str], Allow] = {
    ("VAR_B", "pm_mod1", "pm_mod2"): Allow(BYPASS, _OTHER_TANK, lab_item="tx_pm_coupling"),
    ("TX_RAW", "drv_pad", "tx_bpf"): Allow(PAD, ("the band-pass's output tap meets the matched resistive driver pad: the pad's s11 row proves its input is the "
                                                  "band-pass's load model, and its own S-parameters do not depend on the source, so the cascade is the product")),
}


# --------------------------------------------------------------------------- building


def _build(name: str, answers: dict[str, str], workdir: Path) -> CircuitIR:
    ir = CircuitIR(project=ProjectMeta(id=name, name=name, workdir=str(workdir)))
    for key, value in answers.items():
        ir.requirements.requirements.append(_answer_requirement(key, value))
    tpl = _bench_template(name)
    if tpl is None:
        plan = templates_mod.design_from_requirements(ir, _REAL, confirmed=True)
    else:
        inputs, unusable = read_inputs(ir)
        plan = tpl.build(ir, inputs, unusable, _REAL, confirmed=True)
        assert not plan.buildable or add_board(tpl, ir, plan, confirmed=True) is None
    assert plan is not None and plan.buildable, (name, plan.notes if plan else None)
    Orchestrator.apply_proposals(ir, [IRProposal(description=c.description, target=c.target, operation=c.operation, payload=c.payload) for c in plan.changes])
    return ir


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, CircuitIR]:
    """Every build, confirmed and applied once for the module's read-only tests (a test that changes an IR copies it)."""
    if not HAS_LIBS:
        pytest.skip("KiCad 10 libraries with the RF parts not installed (set KICAD10_SYMBOL_DIR)")
    return {name: _build(name, answers, tmp_path_factory.mktemp(name)) for name, answers in BUILDS.items()}


# --------------------------------------------------------------------------- the allowance checks


def _pins(ir: CircuitIR) -> dict[str, list[tuple[str, str]]]:
    """ref -> [(pin, net)]."""
    out: dict[str, list[tuple[str, str]]] = {}
    for n in ir.nets:
        for p in n.pins:
            out.setdefault(p.component_ref, []).append((p.pin_number, n.name))
    return out


def _other_net(ir: CircuitIR, ref: str, net: str) -> str:
    nets = [x for _, x in _pins(ir)[ref]]
    assert len(nets) == 2 and net in nets, (ref, nets)
    return nets[1] if nets[0] == net else nets[0]


def _value(ir: CircuitIR, ref: str) -> float:
    """A part's value as the fixtures simulate it: a network binding's, else its design-deck binding's, else (a part the deck leaves out as a
    dead branch) its value text read by the part-value parser (5 significant digits: enough for the estimates here)."""
    assert ir.rf is not None
    for nw in ir.rf.networks:
        b = nw.bindings.get(ref)
        if b is not None and b.value is not None:
            return float(b.value.value)
    c = ir.component(ref)
    assert c is not None, ref
    if c.spice is not None and c.spice.value is not None:
        return float(c.spice.value.value)
    value = parse_part_value(c.value, {"R": "ohm", "C": "F", "L": "H"}[ref[0]])
    assert value is not None, (ref, c.value)
    return value


def _row_frequencies(nw: RFNetwork) -> list[float]:
    return [float(t.value) for e in (*nw.expectations, *nw.probes) for t in (e.at, e.ref_at) if t is not None]


def _is_ac_ground(ir: CircuitIR, net: str) -> bool:
    kinds = {n.name: n.kind for n in ir.nets}
    return net == GROUND or kinds.get(net) is NetKind.POWER


def _port_on(nw: RFNetwork, net: str):
    return next((p for p in nw.ports if p.net == net and p.kind in ("port", "probe")), None)


def _matched_pad(nw: RFNetwork, net: str) -> bool:
    """A network of resistors only with an at_most s11 row driven at its port on ``net``."""
    port = _port_on(nw, net)
    return (port is not None and all(PASSIVE_REF.match(m) and m.startswith("R") for m in nw.members)
            and any(e.quantity == "s11_db" and e.bound == "at_most" and e.drive == port.name for e in nw.expectations))


def _check_port_model(ir: CircuitIR, nw: RFNetwork, net: str, allow: Allow, all_allowed: list[str]) -> list[str]:
    problems: list[str] = []
    port = _port_on(nw, net)
    if port is None or port.kind != "port":
        return [f"{nw.id}: no port on {net} whose model could include the part"]
    note = port.z0_ohm.provenance.note or ""
    m = MODEL_KEY_IN_NOTE.search(note)
    key = m.group(1) if m is not None else ""
    assert ir.rf is not None
    if not key or key not in ir.rf.model_values:
        problems.append(f"{nw.id}.{port.name}: the port's resistance is not a listed model value ({key!r})")
    if allow.phrase not in note:
        problems.append(f"{nw.id}.{port.name}: the port model's text does not say {allow.phrase!r} (the fixture records {note[:120]!r})")
    grounding = model_grounding_result(ir)
    rows = {r["key"]: r for r in (grounding.details["rows"] if grounding is not None else [])}
    if allow.phrase not in str(rows.get(key, {}).get("note", "")):
        problems.append(f"rf.model_grounding does not name {key!r} with {allow.phrase!r}")
    conductance = 0.0
    for ref in all_allowed:
        if not ref.startswith("R") or not _is_ac_ground(ir, _other_net(ir, ref, net)):
            problems.append(f"{ref}: a port model can include a resistor from the port net to an ac ground, not {ref}")
            continue
        conductance += 1.0 / _value(ir, ref)
    if conductance and not float(port.z0_ohm.value) <= 1.0 / conductance * (1 + 1e-12):
        problems.append(f"{nw.id}.{port.name}: {float(port.z0_ohm.value):g} ohm exceeds the {1.0 / conductance:g} ohm of the parts it is said to include")
    return problems


def _solve(elements: list[tuple[str, str, complex]], inject: dict[str, complex], grounds: set[str]) -> dict[str, complex]:
    """Node voltages of a linear AC circuit of two-terminal admittances with currents injected into its nodes (``grounds`` at 0 V)."""
    nets = sorted({n for a, b, _ in elements for n in (a, b)} - grounds)
    at = {n: i for i, n in enumerate(nets)}
    m = [[0j] * (len(nets) + 1) for _ in nets]
    for a, b, y in elements:
        for x, o in ((a, b), (b, a)):
            if x in at:
                m[at[x]][at[x]] += y
                if o in at:
                    m[at[x]][at[o]] -= y
    for net, i in inject.items():
        m[at[net]][-1] += i
    for k in range(len(nets)):  # Gauss-Jordan with partial pivoting
        piv = max(range(k, len(nets)), key=lambda r: abs(m[r][k]))
        m[k], m[piv] = m[piv], m[k]
        for r in range(len(nets)):
            if r != k and m[r][k] != 0:
                q = m[r][k] / m[k][k]
                m[r] = [x - q * y for x, y in zip(m[r], m[k])]
    return {**{g: 0j for g in grounds}, **{n: m[at[n]][-1] / m[at[n]][at[n]] for n in nets}}


def _admittance(ir: CircuitIR, nw: RFNetwork, ref: str, state: str | None, w: float) -> tuple[str, str, complex]:
    """A member of ``nw`` as its fixture simulates it, a two-terminal admittance at ``w``: R / C at their value (the state's binding, else the
    network's, else the part's), an inductor with its ``loss_q`` series resistor (2 pi q_ref_hz L / Q), a varactor (a diode on the
    ``model.varactor`` card, anode on GND) as its junction capacitance CJO / (1 + V/VJ)^M at the reverse bias the state's control port sets
    through the network's R / L members (no DC current flows there). Anything else fails: the solve would not be the fixture's circuit."""
    a, b = [n for _, n in _pins(ir)[ref]]
    st = nw.state(state) if state is not None else None
    binding = (st.bindings.get(ref) if st is not None else None) or nw.bindings.get(ref)
    if ref.startswith("D"):
        card = binding.model_card if binding is not None else None
        assert card is not None and "model.varactor" in (card.provenance.note or "") and GROUND in (a, b), (nw.id, ref)
        reach, todo = {b if a == GROUND else a}, [b if a == GROUND else a]
        while todo:  # the DC path from the cathode through the network's resistors and inductors
            n = todo.pop()
            for r in nw.members:
                if r[0] in "RL" and PASSIVE_REF.match(r) and n in (nets := [x for _, x in _pins(ir)[r]]):
                    for x in nets:
                        if x not in reach and x != GROUND:
                            reach.add(x)
                            todo.append(x)
        control = [p for p in nw.ports if p.kind == "control" and p.net in reach]
        assert st is not None and len(control) == 1 and control[0].name in st.port_dc_v, (nw.id, ref, state)
        p = ir.parameters
        c = varactor_c_f(p["model.varactor.cjo"].value, p["model.varactor.vj"].value, p["model.varactor.m"].value,
                         float(st.port_dc_v[control[0].name].value))
        return a, b, 1j * w * c
    value = float(binding.value.value) if binding is not None and binding.value is not None else _value(ir, ref)
    if ref.startswith("R"):
        return a, b, 1.0 / value
    if ref.startswith("C"):
        return a, b, 1j * w * value
    assert ref.startswith("L"), (nw.id, ref)
    q = nw.loss_q.get(ref)
    r_loss = 2 * math.pi * float(nw.q_ref_hz.value) * value / float(q.value) if q is not None and nw.q_ref_hz is not None else 0.0
    return a, b, 1.0 / (r_loss + 1j * w * value)


def _terminations(nw: RFNetwork, drive: str | None, done: set[str]) -> tuple[list[tuple[str, str, complex]], dict[str, complex], set[str]]:
    """``nw``'s ports as its fixture terminates them (the port nets in ``done`` are left alone): a ``port`` its z0 to GND - the drive also
    a 1 V source behind it (its Norton current) -, a ``control`` / ``rail`` port an AC ground, a probe nothing."""
    elements: list[tuple[str, str, complex]] = []
    inject: dict[str, complex] = {}
    grounds = {GROUND}
    for port in nw.ports:
        if port.kind in ("control", "rail"):
            grounds.add(port.net)
        elif port.kind == "port" and port.net not in done:
            y = 1.0 / float(port.z0_ohm.value)
            elements.append((port.net, GROUND, y))
            if port.name == drive:
                inject[port.net] = y
            done.add(port.net)
    return elements, inject, grounds


def _circuit(ir: CircuitIR, nw: RFNetwork, state: str | None, w: float, drive: str | None, hung: tuple[RFNetwork, list[str]] | None = None):
    """(elements, injected currents, grounds) of ``nw``'s fixture circuit driven at ``drive`` - with ``hung``'s parts (each as its own
    network simulates it in the same state) hung on passively, that network's ports at their port models."""
    elements = [_admittance(ir, nw, ref, state, w) for ref in nw.members]
    done: set[str] = set()
    ports, inject, grounds = _terminations(nw, drive, done)
    elements += ports
    if hung is not None:
        other, parts = hung
        elements += [_admittance(ir, other, ref, state, w) for ref in parts]
        more, _, g = _terminations(other, None, done)
        elements += more
        grounds |= g
    return elements, inject, grounds


def _row_value(ir: CircuitIR, nw: RFNetwork, e, hung: tuple[RFNetwork, list[str]] | None = None) -> float:
    """A ``phase21_deg`` / ``s21_db`` row of ``nw`` from the linear nodal solve of its fixture circuit (a 1 V source behind the drive's z0,
    S21 = 2 V_to / V_s sqrt(R_drive / R_to), a probe only by its phase)."""
    w = 2 * math.pi * float(e.at.value)
    elements, inject, grounds = _circuit(ir, nw, e.state, w, e.drive, hung)
    v = _solve(elements, inject, grounds)[nw.port(e.to).net]
    if e.quantity == "phase21_deg":
        return math.degrees(cmath.phase(v))
    assert e.quantity == "s21_db" and nw.port(e.to).kind == "port", (nw.id, e.id, e.quantity)
    return 20 * math.log10(2 * abs(v) * math.sqrt(float(nw.port(e.drive).z0_ohm.value) / float(nw.port(e.to).z0_ohm.value)))


def _check_bypass(ir: CircuitIR, nw: RFNetwork, other: RFNetwork, net: str, refs: list[str]) -> list[str]:
    """``refs`` (inductors of ``other``) hang on ``net``, which both networks hold a shared capacitor to GND on (module docstring: a bypass).

    In every state of ``nw`` at its reference frequency, the branch the bypass grounds is the outsiders and the rest of ``other`` behind
    them (the members ``nw`` does not hold, ``other``'s ports at their port models); the shared capacitor's impedance must be at most
    :data:`BYPASS_SHARE` of that branch's, and ``nw``'s rows with the branch hung on ``net`` passively must move by at most
    :data:`JOINT_TOL_SHARE` of their tolerance."""
    on = net_refs(ir)[net]
    shared = [r for r in on & set(nw.members) & set(other.members) if r.startswith("C") and _other_net(ir, r, net) == GROUND]
    if not shared:
        return [f"{net}: {nw.id} and {other.id} hold no shared capacitor to GND there"]
    branch = [r for r in other.members if r not in nw.members]
    problems = [f"{ref}: a bypass allowance covers an inductor of the other network hanging on the node, not {ref}"
                for ref in refs if not ref.startswith("L") or ref not in branch]
    if problems:
        return problems
    f = float((nw.q_ref_hz or other.q_ref_hz).value) if (nw.q_ref_hz or other.q_ref_hz) is not None else min(_row_frequencies(nw))
    w = 2 * math.pi * f
    z_c = 1.0 / (1j * w * sum(_value(ir, r) for r in shared))
    for state in [s.id for s in nw.states] or [None]:
        if state is not None and other.state(state) is None:
            problems.append(f"{other.id} has no state {state!r} of {nw.id}: the branch's varactors have no bias to be solved at")
            continue
        elements = [_admittance(ir, other, ref, state, w) for ref in branch]
        ports, _, grounds = _terminations(other, None, set())
        z_branch = _solve(elements + ports, {net: 1.0}, grounds)[net]
        if not abs(z_c) <= BYPASS_SHARE * abs(z_branch):
            problems.append(f"{net} ({state}): the bypass's {abs(z_c):.4g} ohm is {abs(z_c) / abs(z_branch):.2%} of the {abs(z_branch):.4g} ohm branch "
                            f"into {', '.join(refs)} and {other.id} at {f:.6g} Hz, more than {BYPASS_SHARE:.0%}")
        for e in nw.expectations:
            if e.state != state:
                continue
            tol = float(e.tol_abs.value) if e.tol_abs is not None else None
            if tol is None:
                problems.append(f"{nw.id}.{e.id}: a row without an absolute tolerance is not bounded by the bypass check")
                continue
            moved = _row_value(ir, nw, e, (other, branch)) - _row_value(ir, nw, e)
            if not abs(moved) <= JOINT_TOL_SHARE * tol:
                problems.append(f"{nw.id}.{e.id}: {other.id}'s branch hung on {net} moves the row by {moved:+.4g}, more than {JOINT_TOL_SHARE:g} x its {tol:g}")
    return problems


def _owner(ir: CircuitIR, ref: str, besides: str) -> list[RFNetwork]:
    assert ir.rf is not None
    return [n for n in ir.rf.networks if ref in n.members and n.id != besides]


def _check_pad(ir: CircuitIR, nw: RFNetwork, net: str, ref: str) -> list[str]:
    port = _port_on(nw, net)
    if port is None or port.kind != "port":
        return [f"{nw.id}: no port on {net}"]
    for other in _owner(ir, ref, nw.id):
        o_port = _port_on(other, net)
        if o_port is None or o_port.kind != "port" or float(o_port.z0_ohm.value) != float(port.z0_ohm.value):
            continue
        if _matched_pad(other, net) or _matched_pad(nw, net):
            return []
    return [f"{nw.id} / {ref} on {net}: neither side is a matched resistive pad with the other's port resistance"]


def _check_dc_block(ir: CircuitIR, nw: RFNetwork, net: str, ref: str) -> list[str]:
    port = _port_on(nw, net)
    if port is None or port.kind != "port":
        return [f"{nw.id}: no port on {net}"]
    if not ref.startswith("C"):
        return [f"{ref}: a DC block is a capacitor"]
    far = _other_net(ir, ref, net)
    behind = net_refs(ir)[far] - {ref}
    if far == GROUND or behind & set(nw.members):
        return [f"{ref}: its far net {far} is GND or holds a member of {nw.id} - not a series block into the modelled part"]
    modelled = [r for r in behind if not PASSIVE_REF.match(r)]
    chokes = [r for r in behind if PASSIVE_REF.match(r)]
    problems = [] if modelled else [f"{ref}: nothing behind it on {far} for the port model to stand for"]
    for r in chokes:
        if not r.startswith("L") or not _is_ac_ground(ir, _other_net(ir, r, far)):
            problems.append(f"{r} behind {ref} on {far} is not a bias choke to a rail")
    f = min(_row_frequencies(nw))
    w, z0 = 2 * math.pi * f, float(port.z0_ohm.value)
    effect = 10 * math.log10(1 + (1.0 / (w * _value(ir, ref)) / (2 * z0)) ** 2)
    effect += sum(10 * math.log10(1 + (z0 / (2 * w * _value(ir, r))) ** 2) for r in chokes if r.startswith("L"))
    if not effect <= DC_BLOCK_DB:
        problems.append(f"{ref} (with {chokes}) changes |S21| by {effect:.4f} dB at {f:.6g} Hz, more than {DC_BLOCK_DB} dB")
    return problems


def _covered(ir: CircuitIR, o: Outsider) -> list[str]:
    """The other networks that hold the outsider together with every part of its network on that net (the junction is inside them)."""
    assert ir.rf is not None
    nw = ir.rf.network(o.network)
    assert nw is not None
    need = (net_refs(ir)[o.net] & set(nw.members)) | {o.ref}
    return sorted(c.id for c in ir.rf.networks if c.id != nw.id and need <= set(c.members))


def _explain_outsiders(ir: CircuitIR) -> tuple[list[str], set[tuple[str, str]], dict[str, list[str]]]:
    """(unexplained problems, the allowance keys used, cover network -> the outsiders it holds)."""
    assert ir.rf is not None
    found = outsiders(ir)
    problems: list[str] = []
    used: set[tuple[str, str]] = set()
    covers: dict[str, list[str]] = {}
    for o in found:
        cover = _covered(ir, o)
        if cover:
            for c in cover:
                covers.setdefault(c, []).append(f"{o.network}:{o.ref}")
            continue
        allow = ALLOWED_OUTSIDERS.get((o.network, o.ref))
        if allow is None:
            problems.append(f"{o.network}: {o.ref} on {o.net} ({'port ' + o.port if o.port else 'internal net'}) is not a member and no allowance covers it")
            continue
        used.add((o.network, o.ref))
        nw = ir.rf.network(o.network)
        assert nw is not None
        if allow.lab_item and allow.lab_item not in {x.id for x in ir.rf.lab_items}:
            problems.append(f"{o.network}: {o.ref}'s allowance rests on the lab item {allow.lab_item!r}, which the design does not carry")
        if allow.kind == PORT_MODEL:
            same = [x.ref for x in found if x.network == o.network and x.net == o.net and ALLOWED_OUTSIDERS.get((x.network, x.ref), allow).kind == PORT_MODEL]
            problems += _check_port_model(ir, nw, o.net, allow, same)
        elif allow.kind == BYPASS:
            others = _owner(ir, o.ref, nw.id)
            problems += [f"{o.ref}: in no other network"] if not others else _check_bypass(ir, nw, others[0], o.net, [o.ref])
        elif allow.kind == PAD:
            problems += _check_pad(ir, nw, o.net, o.ref)
        elif allow.kind == DC_BLOCK:
            problems += _check_dc_block(ir, nw, o.net, o.ref)
        else:  # pragma: no cover - the table's kinds are the four above
            problems.append(f"unknown allowance kind {allow.kind!r}")
    return problems, used, covers


def _explain_junctions(ir: CircuitIR) -> tuple[list[str], set[tuple[str, str, str]], list[Junction]]:
    """(unexplained problems, the allowance keys used, the junctions a nesting or a cover resolves)."""
    assert ir.rf is not None
    problems: list[str] = []
    used: set[tuple[str, str, str]] = set()
    resolved: list[Junction] = []
    for j in junctions(ir):
        if j.nested or j.covered_by:
            resolved.append(j)
            continue
        a, b = sorted(j.networks)
        key = (j.net, a, b)
        allow = ALLOWED_JUNCTIONS.get(key)
        if allow is None:
            problems.append(f"{j.net}: {j.networks[0]} ({', '.join(j.sides[0])}) meets {j.networks[1]} ({', '.join(j.sides[1])}) with no network holding both")
            continue
        used.add(key)
        na, nb = ir.rf.network(a), ir.rf.network(b)
        assert na is not None and nb is not None
        if allow.lab_item and allow.lab_item not in {x.id for x in ir.rf.lab_items}:
            problems.append(f"{j.net}: the junction's allowance rests on the lab item {allow.lab_item!r}, which the design does not carry")
        side = dict(zip(j.networks, j.sides))
        if allow.kind == BYPASS:
            problems += _check_bypass(ir, na, nb, j.net, list(side[b])) + _check_bypass(ir, nb, na, j.net, list(side[a]))
        elif allow.kind == PAD:
            pa, pb = _port_on(na, j.net), _port_on(nb, j.net)
            same_z = pa is not None and pb is not None and pa.kind == pb.kind == "port" and float(pa.z0_ohm.value) == float(pb.z0_ohm.value)
            if not same_z or not (_matched_pad(na, j.net) or _matched_pad(nb, j.net)):
                problems.append(f"{j.net}: {a} / {b} is not a matched resistive pad beside a port of the same resistance")
        else:
            problems.append(f"{j.net}: a junction allowance of kind {allow.kind!r} is not checked here")
    return problems, used, resolved


# --------------------------------------------------------------------------- always (no library read)


def test_the_allowance_tables_name_known_kinds_and_sorted_junctions() -> None:
    assert {a.kind for a in ALLOWED_OUTSIDERS.values()} <= {PORT_MODEL, BYPASS, PAD, DC_BLOCK}
    assert {a.kind for a in ALLOWED_JUNCTIONS.values()} <= {BYPASS, PAD}
    assert all(a < b for _, a, b in ALLOWED_JUNCTIONS) and all(len(a.reason) > 40 for a in [*ALLOWED_OUTSIDERS.values(), *ALLOWED_JUNCTIONS.values()])
    assert all(a.phrase for a in ALLOWED_OUTSIDERS.values() if a.kind == PORT_MODEL)
    assert set(BUILDS) == set(NETWORKS) and {BUILDS[b]["radio_build"] for b in BUILDS} == {
        "audio_ptt", "rx_backend", "rx_frontend", "tx_exciter", "transceiver", "transceiver_conducted"}


# --------------------------------------------------------------------------- with the libraries


@needs_libs
def test_every_build_holds_its_fixture_networks(built: dict[str, CircuitIR]) -> None:
    for name, ir in built.items():
        ids = sorted(n.id for n in ir.rf.networks) if ir.rf is not None else []
        assert ids == NETWORKS[name], name


@needs_libs
@pytest.mark.parametrize("build", list(BUILDS))
def test_no_fixture_network_leaves_a_part_of_its_circuit_out(built: dict[str, CircuitIR], build: str) -> None:
    ir = built[build]
    if ir.rf is None or not ir.rf.networks:
        assert NETWORKS[build] == []
        return
    problems, _, covers = _explain_outsiders(ir)
    assert problems == []
    # the outsiders a cascade or an enclosing network resolves, per build: nothing else resolves one
    expected = next((v for prefix, v in (("tx_exciter", {"pa_lpf"}), ("rx_backend", {"if1_filter"}), ("transceiver", {"ant_end", "if1_filter"}))
                     if build.startswith(prefix)), set())
    assert set(covers) == expected, covers


@needs_libs
@pytest.mark.parametrize("build", list(BUILDS))
def test_no_two_networks_meet_tap_to_tap(built: dict[str, CircuitIR], build: str) -> None:
    ir = built[build]
    if ir.rf is None or not ir.rf.networks:
        return
    problems, _, resolved = _explain_junctions(ir)
    assert problems == []
    # every junction a third network resolves is a cascade fixture's inside (or the ladder inside its filter)
    for j in resolved:
        assert j.nested or set(j.covered_by) <= {"pa_lpf", "ant_end", "if1_filter"}, j


@needs_libs
def test_every_allowance_is_used_on_some_build(built: dict[str, CircuitIR]) -> None:
    used_o: set[tuple[str, str]] = set()
    used_j: set[tuple[str, str, str]] = set()
    for ir in built.values():
        if ir.rf is None or not ir.rf.networks:
            continue
        used_o |= _explain_outsiders(ir)[1]
        used_j |= _explain_junctions(ir)[1]
    assert set(ALLOWED_OUTSIDERS) - used_o == set()
    assert set(ALLOWED_JUNCTIONS) - used_j == set()


@needs_libs
def test_the_front_end_fixtures_hold_the_lna_and_post_amp_parts(built: dict[str, CircuitIR]) -> None:
    """The four defects the wave-2 review left: the LNA's divider and choke, the post-amp's divider (fixed by membership), the feed decouplings."""
    for name in ("rx_frontend", "transceiver"):
        ir = built[name]
        rf = ir.rf
        assert rf is not None
        members = {n.id: set(n.members) for n in rf.networks}
        assert {"R601", "R602"} <= members["fe_bpf2"] and {"L603", "R604", "C607"} <= members["fe_bpf3"]
        assert {"R651", "R652"} <= members["diplexer"]
        assert {"C704"} <= members["lo_tank1"] and {"C711"} <= members["lo_tank2"] and {"C718"} <= members["lo_bpf"]
        for nid in ("fe_bpf2", "fe_bpf3", "diplexer"):
            rails = [p for p in rf.network(nid).ports if p.kind == "rail"]
            assert [(p.net, float(p.voltage_v.value)) for p in rails] == [("RX_5V", 5.0)], nid
        assert rf.network("fe_bpf3").loss_q["L603"].value == ir.parameters["model.l_q.uhf"].value
        p = ir.parameters
        # fe_bpf2 is designed into the port model beside the divider, and its row reads the port model's share
        assert p["fe.lna.r_div"].value == pytest.approx(3300 * 2200 / 5500)
        assert p["fe_bpf2.r_load_eff"].value == pytest.approx(50 * 1320 / 1370)
        assert p["fe_bpf2.load_share_db"].value == pytest.approx(10 * math.log10(1320 / 1370))
        assert p["fe_bpf2.s21"].value == pytest.approx(p["fe_bpf2.s21_net"].value + p["fe_bpf2.load_share_db"].value)
        assert p["fe_bpf2.s21"].value == pytest.approx(-3.5285, abs=1e-3) and p["fe_bpf2.rel_image"].value == pytest.approx(-13.565, abs=1e-3)
        # fe_bpf3's input tap absorbs the choke: the collector port with L603 (100 nH, Q 40) at f_c
        assert p["fe_bpf3.port_r"].value == pytest.approx(48.2683, abs=1e-3) and p["fe_bpf3.port_x"].value == pytest.approx(8.5389, abs=1e-3)
        assert p["fe_bpf3.s21"].provenance.tool == "calc.rf.resonator.top_c.ported_s21_db"
        assert p["fe_bpf3.s21"].value == pytest.approx(-9.322, abs=1e-3) and p["fe_bpf3.rel_lo1"].value == pytest.approx(-15.426, abs=1e-3)
    for name in ("tx_exciter", "transceiver"):
        members = {n.id: set(n.members) for n in built[name].rf.networks}
        assert {"C814"} <= members["tx_tank1"] and {"C821"} <= members["tx_tank2"] and {"C828"} <= members["tx_bpf"]


@needs_libs
def test_the_pm_port_models_state_what_they_include(built: dict[str, CircuitIR]) -> None:
    """pm_mod1 / pm_mod2 leave the followers' dividers and the first emitter resistor outside by a stated port model: recorded and named."""
    for name in ("tx_exciter", "transceiver"):
        ir = built[name]
        rf = ir.rf
        assert rf is not None
        p = ir.parameters
        for nid, port, phrase in (("pm_mod1", "buf", DIVIDER_INSIDE), ("pm_mod2", "buf", DIVIDER_INSIDE), ("pm_mod2", "src", EMITTER_R_INSIDE)):
            z0 = rf.network(nid).port(port).z0_ohm
            assert phrase in (z0.provenance.note or ""), (nid, port)
        rows = {r["key"]: r for r in model_grounding_result(ir).details["rows"]}
        assert DIVIDER_INSIDE in rows["model.buf.r_in"]["note"] and EMITTER_R_INSIDE in rows["model.buf.r_out"]["note"]
        assert rows["model.buf.r_in"]["status"] == S.NOT_VERIFIED.value and rows["model.buf.r_out"]["status"] == S.NOT_VERIFIED.value
        for k in (1, 2):  # a port model that includes the divider cannot exceed it; the emitter resistor likewise
            div = 1.0 / (1.0 / p[f"tx.buf{k}.r_b1"].value + 1.0 / p[f"tx.buf{k}.r_b2"].value)
            assert p["model.buf.r_in"].value <= div == pytest.approx(10312.5)
        assert p["model.buf.r_out"].value <= p["tx.buf1.r_e"].value


def _pm_rows(ir: CircuitIR, nid: str) -> dict[str, object]:
    """state -> the phase21 row of a PM network."""
    nw = ir.rf.network(nid)
    return {e.state: e for e in nw.expectations if e.quantity == "phase21_deg"}


def _driven_pm(ir: CircuitIR, state: str, gain: float) -> tuple[float, float]:
    """(tank 1's phase against the TCXO, tank 2's against its own drive) in ``state`` with both tanks on the shared bias node and follower 1
    a source of ``gain`` times its input behind ``model.buf.r_out`` (the pm_mod2 source port) - the joint circuit no fixture holds (a
    fixture has no controlled source), solved by superposition: the TCXO alone, and a unit current into the follower's output node."""
    rf = ir.rf
    m1, m2 = rf.network("pm_mod1"), rf.network("pm_mod2")
    e1, e2 = _pm_rows(ir, "pm_mod1")[state], _pm_rows(ir, "pm_mod2")[state]
    w = 2 * math.pi * float(e1.at.value)
    elements, inject, grounds = _circuit(ir, m1, state, w, e1.drive, (m2, [r for r in m2.members if r not in m1.members]))
    src2, into = m2.port(e2.drive), m1.port("buf").net  # follower 1: its input is pm_mod1's load port, its output pm_mod2's source port
    y2 = 1.0 / float(src2.z0_ohm.value)
    a = _solve(elements, inject, grounds)
    t = _solve(elements, {src2.net: y2}, grounds)  # a 1 V emf behind the follower's output resistance
    v_in = a[into] / (1 - gain * t[into])
    emf = gain * v_in
    tank1 = a[m1.port(e1.to).net] + emf * t[m1.port(e1.to).net]
    tank2 = a[m2.port(e2.to).net] + emf * t[m2.port(e2.to).net]
    return math.degrees(cmath.phase(tank1)), math.degrees(cmath.phase(tank2 / emf))


@needs_libs
def test_the_pm_tanks_couple_through_their_shared_bias_node_as_the_texts_and_the_lab_item_say(built: dict[str, CircuitIR]) -> None:
    """The two tanks' inductors return to VAR_B. Passively the other tank is a series-resonant branch of about 14-15 ohm there (not the
    inductor's 142 ohm), the 0.43 ohm bypass about 3 % of it, and it moves a fixture's phases by at most 0.032 deg. Driven - follower 1
    feeds tank 2 with tank 1's own signal - tank 1 moves by +0.70 to +1.04 deg and tank 2, against its own drive, by +2.85 deg in every
    state (the followers as unity-gain sources behind model.buf.r_out), while the chord of the two tanks moves by about 0.03 deg: the
    figures tx_chain / the theory text / the lab item tx_pm_coupling state (a linear estimate, not a verdict)."""
    from ai_eda.design.rf.blocks import tx_chain
    from ai_eda.design.rf.t_tx_exciter import Kr447TxExciterTemplate

    doc = " ".join((tx_chain.__doc__ or "").split())
    for words in ("about 14-15 ohm", "about 3 % of it", "at most 0.032 deg", "+0.70 to +1.04 deg", "+2.85 deg in every state", "about 0.03 deg of 79.4 deg", "tx_pm_coupling"):
        assert words in doc, words
    assert "0.3 %" not in doc
    for name in ("tx_exciter", "transceiver"):
        ir = built[name]
        rf = ir.rf
        assert rf is not None
        lab = {x.id: x for x in rf.lab_items}["tx_pm_coupling"]
        assert lab.block == "tx_mod" and "VAR_B" in lab.what and "+2.85 deg" in lab.what and "phase21 rows hold for the tank alone" in lab.reason
        m1, m2 = rf.network("pm_mod1"), rf.network("pm_mod2")
        w = 2 * math.pi * float(ir.parameters["tx.f_ref"].value)
        z_c = abs(1.0 / (w * float(ir.parameters["pm.c_bypass"].value)))
        shift1, shift2, passive, branch = {}, {}, [], []
        for state in ("bias_lo", "bias_nom", "bias_hi"):
            e1, e2 = _pm_rows(ir, "pm_mod1")[state], _pm_rows(ir, "pm_mod2")[state]
            alone1, alone2 = _row_value(ir, m1, e1), _row_value(ir, m2, e2)
            # the solve is the fixture's network: the calculator's exact nominal (0.003 deg apart: ngspice reads the netlist's value, as here)
            assert alone1 == pytest.approx(float(e1.nominal.value), abs=0.005) and alone2 == pytest.approx(float(e2.nominal.value), abs=0.005)
            for nw, other in ((m1, m2), (m2, m1)):
                parts = [r for r in other.members if r not in nw.members]
                elements = [_admittance(ir, other, r, state, w) for r in parts]
                ports, _, grounds = _terminations(other, None, set())
                branch.append(abs(_solve(elements + ports, {"VAR_B": 1.0}, grounds)["VAR_B"]))
                e = _pm_rows(ir, nw.id)[state]
                passive.append(abs(_row_value(ir, nw, e, (other, parts)) - _row_value(ir, nw, e)))
            tank1, tank2 = _driven_pm(ir, state, 1.0)
            shift1[state], shift2[state] = tank1 - alone1, tank2 - alone2
        assert z_c == pytest.approx(0.4267, abs=1e-3) and 14.0 < min(branch) and max(branch) < 15.5
        assert 0.027 < z_c / max(branch) and z_c / min(branch) < 0.031  # about 3 %, not the 0.3 % of the inductor's reactance
        assert max(passive) == pytest.approx(0.032, abs=5e-4)
        assert min(shift1.values()) == pytest.approx(0.70, abs=0.005) and max(shift1.values()) == pytest.approx(1.04, abs=0.005)
        assert all(v == pytest.approx(2.85, abs=0.005) for v in shift2.values())
        chord = (shift1["bias_hi"] - shift1["bias_lo"]) + (shift2["bias_hi"] - shift2["bias_lo"])
        whole = sum(float(_pm_rows(ir, n)["bias_hi"].nominal.value) - float(_pm_rows(ir, n)["bias_lo"].nominal.value) for n in ("pm_mod1", "pm_mod2"))
        assert abs(chord) == pytest.approx(0.03, abs=0.01) and whole == pytest.approx(79.4, abs=0.05)
    # the theory text says the same
    text = " ".join(s.body for s in Kr447TxExciterTemplate().theory(built["tx_exciter"]) if hasattr(s, "body"))
    assert "0.3 %" not in text and "약 3 %" in text and "tx_pm_coupling" in text


@needs_libs
def test_a_port_model_says_whether_its_bias_parts_are_inside_it_and_nothing_counts_twice(built: dict[str, CircuitIR]) -> None:
    """A port model either includes the bias parts on its net (``model.buf.*``: no member for them) or is the transistor's own resistance with
    those parts as fixture members beside it (``model.bfr92.*``, ``model.lna.port_r``, ``model.ifamp.port_r``) - its text says which, so a
    value grounded later by a measurement is measured with or without them, and no fixture counts a divider or a choke twice."""
    seen: set[str] = set()
    for name, ir in built.items():
        if ir.rf is None or not ir.rf.networks:
            continue
        rows = {r["key"]: r for r in model_grounding_result(ir).details["rows"]}
        on = net_refs(ir)
        for nw in ir.rf.networks:
            for port in nw.ports:
                m = MODEL_KEY_IN_NOTE.search(port.z0_ohm.provenance.note or "") if port.kind == "port" and port.z0_ohm is not None else None
                if m is None:
                    continue
                key, note = m.group(1), port.z0_ohm.provenance.note or ""
                passives = sorted(r for r in on[port.net] if PASSIVE_REF.match(r))
                if key in TRANSISTOR_PORTS:
                    seen.add(key)
                    assert all(w in note and w in rows[key]["note"] for w in BIAS_BESIDE), (name, nw.id, port.name, note[:160])
                    assert "inside this port model" not in note, (name, nw.id, port.name)
                    # the parts beside the port are members, as the text says
                    assert [r for r in passives if r not in nw.members] == [], (name, nw.id, port.name, passives)
                if "inside this port model" in note:
                    # a port model that includes a resistor to an ac ground has no member for it (the fixture would count it twice)
                    inside = [r for r in on[port.net] if r.startswith("R") and PASSIVE_REF.match(r) and _is_ac_ground(ir, _other_net(ir, r, port.net))]
                    assert inside and not set(inside) & set(nw.members), (name, nw.id, port.name, inside)
    assert seen == set(TRANSISTOR_PORTS)


@needs_libs
def test_the_cascades_hold_their_component_networks(built: dict[str, CircuitIR]) -> None:
    tx = built["tx_exciter"].rf
    assert tx is not None
    pa_lpf = tx.network("pa_lpf")
    assert set(pa_lpf.members) == set(tx.network("pa_match").members) | set(tx.network("lpf").members)
    assert [(p.name, p.net) for p in pa_lpf.ports] == [("pa_out", "PA_OUT"), ("lpf_out", "TX_OUT"), ("pa_5v", "PA_5V")]
    assert {e.id: e.bound for e in pa_lpf.expectations} == {"s21_fc": "at_least", "s11_fc": "at_most", "s21_2fc": "at_most", "s21_3fc": "at_most"}
    assert built["tx_exciter"].parameters["pa_lpf.s21_min"].value == pytest.approx(-2.0)
    for name, parts, feed_z in (("transceiver", ["pa_match", "trsw", "lpf", "fe_bpf2"], 50.0), ("transceiver_conducted", ["pa_match", "trsw", "lpf", "fe_bpf2"], 50.0),
                                ("transceiver_antenna_match", ["pa_match", "trsw", "lpf", "ant_match", "fe_bpf2"], 36.0)):
        ir = built[name]
        rf = ir.rf
        assert rf is not None
        ant = rf.network("ant_end")
        assert ant.block is None and set(ant.members) == set().union(*(rf.network(n).members for n in parts)), name
        assert [s.id for s in ant.states] == ["tx", "rx"] and ant.states == rf.network("trsw").states
        assert {p.name: (p.net, p.kind) for p in ant.ports} == {"pa_out": ("PA_OUT", "port"), "feed": ("ANT_FEED", "port"), "lna_in": ("LNA_IN", "port"),
                                                                "pa_5v": ("PA_5V", "rail"), "pin_bias": ("PIN_FEED", "rail"), "rx_5v": ("RX_5V", "rail")}
        assert ant.port("feed").z0_ohm.value == feed_z
        match = -0.5 if "match" in name else 0.0
        assert ir.parameters["ant_end.tx_s21_min"].value == pytest.approx(-2.5 + match)
        assert ir.parameters["ant_end.rx_s21_min"].value == pytest.approx(-2.0 + match + ir.parameters["fe_bpf2.s21"].value)
        assert ir.parameters["ant_end.rx_image_max"].value == pytest.approx(ir.parameters["fe_bpf2.rel_image"].value + 1.0)


@needs_libs
def test_the_audit_finds_a_missing_member_and_an_uncovered_junction(built: dict[str, CircuitIR]) -> None:
    """The rules bite: the wave-2 defects put back (the LNA divider out of fe_bpf2, the pa_lpf cascade removed) are found and not explained."""
    ir = built["rx_frontend"].model_copy(deep=True)
    rf = ir.rf
    assert rf is not None
    fe = rf.network("fe_bpf2")
    rf.networks = [n.model_copy(update={"members": [m for m in n.members if m not in ("R601", "R602")]}) if n is fe else n for n in rf.networks]
    found = {(o.network, o.ref) for o in outsiders(ir)}
    assert {("fe_bpf2", "R601"), ("fe_bpf2", "R602")} <= found
    assert any("R601 on LNA_IN" in x for x in _explain_outsiders(ir)[0])
    tx = built["tx_exciter"].model_copy(deep=True)
    assert tx.rf is not None
    tx.rf.networks = [n for n in tx.rf.networks if n.id != "pa_lpf"]
    problems = _explain_junctions(tx)[0]
    assert any(x.startswith("LPF_IN: pa_match") or x.startswith("LPF_IN: lpf") for x in problems), problems
    assert any("C910" in x or "C909" in x for x in _explain_outsiders(tx)[0])
    # an internal node's part is found too (the feed decoupling taken out of lo_tank1)
    lo = built["rx_frontend"].model_copy(deep=True)
    assert lo.rf is not None
    lo.rf.networks = [n.model_copy(update={"members": [m for m in n.members if m != "C704"]}) if n.id == "lo_tank1" else n for n in lo.rf.networks]
    assert Outsider("lo_tank1", "LO_X3_VC", "C704", None) in outsiders(lo)
    assert signal_nets(lo, lo.rf.network("lo_tank1"))["LO_X3_VC"] is None


# --------------------------------------------------------------------------- with the libraries and ngspice


def _run(ir: CircuitIR, ids: list[str], tmp_path: Path) -> dict[str, object]:
    assert ir.rf is not None
    design = ir.rf.model_copy(update={"networks": [ir.rf.network(i) for i in ids]})
    return {r.check_id: r for r in spice_rf_results(ir, {"spice": runner}, tmp_path, design=design)}


@needs_libs
@needs_ngspice
def test_the_changed_fixtures_measure_on_ngspice(built: dict[str, CircuitIR], tmp_path: Path) -> None:
    """Every row of the changed networks PASS; the exact rows read their nominals, the cascades their measured values (ngspice-42)."""
    measured: dict[str, float] = {}
    for name, ids in (("rx_frontend", ["fe_bpf2", "fe_bpf3", "diplexer"]), ("tx_exciter", ["pa_lpf"]), ("transceiver", ["ant_end"])):
        got = _run(built[name], ids, tmp_path / name)
        assert all(r.status is S.PASS for r in got.values()), [(k, r.status, r.message) for k, r in got.items() if r.status is not S.PASS]
        for k, r in got.items():
            if "measured" in r.details:
                measured[k] = r.details["measured"]
                if r.details.get("tolerance") is not None:  # an exact-network row: the netlist realises the designed network
                    assert r.details["measured"] == pytest.approx(r.details["nominal"], abs=1e-3), k
    expect = {
        "spice.rf.fe_bpf2.s21_fc": -3.5285, "spice.rf.fe_bpf2.rel_image": -13.5652, "spice.rf.fe_bpf3.s21_fc": -9.3220,
        "spice.rf.fe_bpf3.rel_image": -33.5971, "spice.rf.fe_bpf3.rel_lo1": -15.4259, "spice.rf.diplexer.s21_if1": -0.4624,
        "spice.rf.diplexer.s11_lo1": -16.598, "spice.rf.diplexer.s11_sum": -22.748,
        "spice.rf.pa_lpf.s21_fc": -1.2311, "spice.rf.pa_lpf.s11_fc": -17.765, "spice.rf.pa_lpf.s21_2fc": -61.90, "spice.rf.pa_lpf.s21_3fc": -93.79,
        "spice.rf.ant_end.tx.tx_s21_fc": -1.6372, "spice.rf.ant_end.tx.tx_s11_fc": -15.857, "spice.rf.ant_end.tx.tx_s21_2fc": -65.80,
        "spice.rf.ant_end.tx.tx_s21_3fc": -97.82, "spice.rf.ant_end.tx.tx_iso_lna": -37.95, "spice.rf.ant_end.rx.rx_s21_fc": -4.8325,
        "spice.rf.ant_end.rx.rx_rel_image": -13.882, "spice.rf.ant_end.rx.rx_iso_pa": -28.88,
    }
    assert set(expect) <= set(measured)
    for k, v in expect.items():
        assert measured[k] == pytest.approx(v, abs=0.02), (k, measured[k])


@needs_libs
@needs_ngspice
def test_the_joint_passive_pm_circuit_on_ngspice_is_the_nodal_solve(built: dict[str, CircuitIR], tmp_path: Path) -> None:
    """pm_mod1 with tank 2's parts as members (its source port at model.buf.r_out, its load at model.buf.r_in, not driven): ngspice reads
    what the nodal solve of the bypass check computes, and the phases move from the fixture's by at most 0.032 deg."""
    ir = built["tx_exciter"]
    rf = ir.rf
    assert rf is not None
    m1, m2 = rf.network("pm_mod1"), rf.network("pm_mod2")
    parts = [r for r in m2.members if r not in m1.members]
    joint = m1.model_copy(update={
        "id": "pm_joint", "members": [*m1.members, *parts], "bindings": {**m2.bindings, **m1.bindings}, "loss_q": {**m2.loss_q, **m1.loss_q},
        "ports": [*m1.ports, m2.port("src").model_copy(update={"name": "src2"}), m2.port("buf").model_copy(update={"name": "buf2"})],
    })
    for network, hung in ((m1, None), (joint, (m2, parts))):
        design = rf.model_copy(update={"networks": [network]})
        got = {r.check_id.rsplit(".", 1)[-1]: r for r in spice_rf_results(ir, {"spice": runner}, tmp_path / network.id, design=design)}
        for e in m1.expectations:
            measured = got[e.id].details["measured"]
            assert measured == pytest.approx(_row_value(ir, m1, e, hung), abs=1e-3), (network.id, e.id)
            assert abs(measured - _row_value(ir, m1, e)) <= 0.0325, (network.id, e.id)


@needs_libs
@needs_ngspice
def test_the_antenna_end_cascade_with_the_match_passes(built: dict[str, CircuitIR], tmp_path: Path) -> None:
    got = _run(built["transceiver_antenna_match"], ["ant_end", "ant_match"], tmp_path)
    assert all(r.status is S.PASS for r in got.values()), [(k, r.status, r.message) for k, r in got.items() if r.status is not S.PASS]
    m = {k: r.details["measured"] for k, r in got.items() if "measured" in r.details}
    assert m["spice.rf.ant_end.tx.tx_s21_fc"] == pytest.approx(-1.711, abs=0.02) and m["spice.rf.ant_end.rx.rx_s21_fc"] == pytest.approx(-4.906, abs=0.02)
    assert m["spice.rf.ant_end.tx.tx_s11_fc"] == pytest.approx(-15.741, abs=0.02) and m["spice.rf.ant_end.tx.tx_s21_2fc"] == pytest.approx(-69.97, abs=0.05)
