"""``spice.xtalk.<victim>``: pairwise coupled-line (CPL) transients of the coupled spans ``si.crosstalk`` found, judged against a budget.

Invariant: tool evidence about the *routed* copper, never an opinion; a
verdict needs a deck that ran, checked itself and a stated budget. The
victims are the nets whose ``si.crosstalk`` row hands off
(:func:`ai_eda.validation.si.crosstalk_scan`: every neighbour computed by the
closed forms, and :func:`plan_victim` can build the decks); each victim's
computed parallel pieces are grouped per aggressor into *spans* (consecutive
pieces of one aggressor along the victim's pad-to-pad path, one layer, one
width; a span is *exact* when it has one gap - else it is lumped at its
smallest gap, an upper bound) and placed on both nets' paths
(:attr:`ai_eda.tools.si.paths.NetPath.edges`). One deck per (victim,
aggressor, span, orientation), compiled by the SPICE netlist compiler:

    S --RSA-- A_drv ==TA== AI \\                 / AO ==TA== A_far --CLA-- 0
                               P (CPL, length L)
    0 --RSV-- V_drv ==TV== VI /                 \\ VO ==TV== V_rx --CLV-- 0

* the aggressor's source steps from 0 to 1 V with its class's edge ``t_r``
  (the 0-100 % linear ramp spice.si's PULSE uses) and is held; ``RSA`` /
  ``CLA`` its class's ``r_drive_ohm`` / ``c_load_f``; the victim's driver
  holds a level through its ``r_drive_ohm`` (``RSV`` to ground), its
  receiver is ``c_load_f`` (``CLV``) - the driver model of
  :mod:`ai_eda.tools.si.driver` for each net;
* ``P`` is ngspice's CPL (:class:`~ai_eda.ir.SpiceDevice` ``P``, written by
  the compiler) with the Maxwell matrices of the span's smallest gap,
  :func:`ai_eda.tools.calc.xtalk.maxwell_from_modes` of the
  ``calc.tline.edge_coupled_microstrip`` modal values (line 1 the victim,
  line 2 the aggressor; equal widths only);
* the ``T`` lines are each net's path before and after the span, lossless,
  Z0 = ``calc.tline.microstrip.z0`` of the net's main width (the lumping
  spice.si uses), t_d the summed delay of the path pieces they stand for;
* a net's driver end is its pad whose pin can drive (``DRIVING_TYPES``);
  when that is not one pad, both orientations run.

A path may hold arc tracks (:attr:`ai_eda.tools.si.paths.PathEdge.mid`): its
pieces carry their exact lengths, so a span is placed, and the lines before
and after it are timed, along the copper as it is, and a span's end is
measured against an arc piece itself (:func:`~ai_eda.ir.arc.point_arc_fraction`),
never against the chord between the piece's ends. A coupled span is two
straight parallel conductors: ``si.crosstalk`` hands off no piece on curved
copper (it names a curved neighbour instead), and :func:`plan_victim`
refuses a piece that would run over an arc piece of either path, naming it
(the victim's row is then NOT_VERIFIED with that reason). A rounded corner
on the way to a span is part of a ``T`` line like every bend (lumped,
:data:`XT_MODEL_OMITS`). The results of a board with an arc track say so in
their deck version (:data:`XT_DECK_ARC_VERSION`, :func:`deck_version`); a
board without one is planned and stamped exactly as before.

Transient: step = min(t_r, the shortest line delay) / :data:`STEPS_PER_EDGE`,
stop = t_r + hold with spice.si's ``HOLD_EDGES`` / ``HOLD_TAUS`` (the longer
of the two lines' delays, the larger (R + Z0) C); more than
:data:`MAX_POINTS` steps is NOT_VERIFIED "deck too fine". **Self-check** (a
choice of this check, not a source rule): ngspice-46's CPL can simulate an
unmatched pair wrongly with no message and no step rule avoids it
(:mod:`ai_eda.tools.spice.ngspice_shared` item 12d, measured 2026-11-13), so
every deck runs twice - at the step and at :data:`STEP_CHECK_FACTOR` times
it, each in a process of its own (item 12f: one CPL deck per process) - and
its peak is used only when both runs succeed, both settle at their DC levels
(the aggressor's far end at 1 V, the victim's receiver at 0 V, within
:data:`SETTLE_TOL` of the step: the 1 mV-per-volt criterion of that
measurement) and their peaks agree within :data:`AGREE_REL` (its 10 %
criterion) plus :data:`AGREE_ABS`; the larger of the two peaks is judged.
Each run may take :data:`RUN_TIMEOUT_S` in the engine.

Measured 2026-11-13 on KiCad 10.0.6's ``ngspice.dll`` (ngspice-46, build Apr
14 2026), the 2-net 4-layer test board of ``tests/test_spice_xtalk.py``
(50-ohm width, s/h 1.5, a 25 mm span, 25 ohm drivers, 5 pF loads, 1 ns
edge), against an L-K-C ladder of the same matrices (500 segments, 1 ohm/m,
the 2026-11-13 CPL measurement's reference method - a reference, not part of
this check, whose segment count no opened source fixes):

* a ``T`` line at a CPL port makes the engine take femtosecond steps
  (205 328 points for ``tran 1.2p 1.2n``, 25 s; the deck of this check did
  not finish ``tran 1.2p 37n`` in 120 s) and drift with no message: with one
  90 ps ``T`` line the victim reads -7.25 % and the aggressor's far end
  1.112 V at 2 ns, where the ladder gives +2.30 % peak and 0.988 V; with a
  1 ns or 2 ns line the same deck ran normally to 5 / 8 ns;
* without ``T`` lines (the CPL's ports at the pads; both nets exactly the
  span) the runs are fast (under 1 s each) and depend on the step. Aggressor
  driven beside the victim's driver: at this check's step (2.957 ps) and at
  2x it the victim's receiver peaks at 2.180 % / 2.157 % and both settle
  within 8.5e-4 - confirmed, 2.2 % above the ladder's 2.132 %; at a 1.2 ps
  step (not this check's) it peaks at 2.79 % (2.47 % at 2.4 ps) and settles
  at 0.9876 V of 1 V where the ladder reaches 1.000014 V - the self-check
  refuses that pair (not settled, the steps 12 % apart). Aggressor driven
  beside the victim's receiver (``tests/test_spice_xtalk.py``'s engine
  test): 2.033 % / 2.022 % at 2.957 / 5.913 ps, settled - confirmed (no
  ladder reference: its ladder, 250 or 500 segments, did not finish 1.5 ns
  in 400 s; cause not found); 2.214 % and 0.9869 V at 1.2 ps.

So a deck with a ``T`` line at a CPL port does not confirm itself on this
engine (it times out or drifts: NOT_VERIFIED with the reason), and a verdict
rests only on a deck that passed its own check; that the check catches every
wrong CPL run is not proved (two step pairs and one ladder comparison).
ngspice-42 is not installed here and was not run.

Judgement at the victim's receiver pad (each orientation's ``CLV`` end, so
both ends when the victim has no single driver): a deck's peak is the
largest |v| over the run per unit aggressor step (the network is linear, the
falling edge mirrors the rising one). A span's peak is its worst
orientation; ``total`` sums the spans (superposition: the absolute sum
bounds any alignment).

* PASS: every deck ran and confirmed itself and ``total`` <= the budget;
* FAIL (``repair: human``, the aggressor, layer, span and gap named): one
  exact span's deck alone exceeds the budget in every orientation it ran
  (its most favourable one - a choice: an orientation that cannot be told
  must not make a FAIL);
* NOT_VERIFIED otherwise, naming why: an exact span exceeds only in some of
  its orientations (the ends that drive, named; which end drives cannot be
  told), only a span lumped at its smallest gap exceeds, only the sum exceeds
  ("aggressors switching together not simulated"), a deck did not run or did
  not confirm itself, no engine, the :data:`MAX_XTALK_DECKS` cap, a missing
  driver value or budget.

**Budget** (:func:`victim_budget`), a fraction of the swing:

* ``si.net_classes[<class>].xtalk_budget_rel`` when the user stated it
  (``user_requirement``, the only provenance the IR accepts) - it takes
  precedence, and it is the only budget of an analog / rf victim (no opened
  source gives a numeric budget for one) and of a receiver whose thresholds
  are per pin;
* otherwise NM / 3 from grounded datasheet levels (owner's decision,
  2026-11-13): budget = min(VOH_drv - VIH_rx, VIL_rx - VOL_drv) / 3 / swing.
  The margins are TI SCEA005 (1997) Figure 1 and p. 3's definitions
  (NMH = VOH(driver) - VIH(receiver), NML = VIL(receiver) - VOL(driver);
  quoted by the research step, ``crosstalk.json``: "which allows a 300-mV
  noise margin"); the 1/3 is E. Bogatin, EDN Rule of Thumb #19 (2014-11-19:
  a noise budget equal to the noise margin split evenly between reflection
  noise, crosstalk and power noise; quoted there: "less than about 5% of the
  signal swing") - neither re-opened in this phase. ``VOH`` / ``VOL`` are the
  victim driver's grounded ``v_oh`` / ``v_ol``, ``VCC`` its ``v_cc``, ``VIH`` /
  ``VIL`` the receiver's ``v_ih`` / ``v_il`` (:data:`ai_eda.parts.datasheet_facts.SI_FACT_KEYS`).
  A component's grounded facts hold one datasheet row per key: ``v_oh`` at
  ``i_oh``, ``v_ol`` at ``i_ol``, both at the supply ``v_cc``; that row is the
  one read, and the details name it (or say no row current was grounded).
  The modelled victim's only load is ``C_load``, which draws no static
  current, so a row at a larger current only lowers VOH / raises VOL (the
  opened SN74LVC1G04 rows of SCES214AF 5.5: VOH 2.9 / 2.4 / 2.3 V at 0.1 / 16
  / 24 mA, VCC 3 V, as ``crosstalk.json`` quotes them) - the margin errs small.

  **swing** (defined once here; a choice, no opened source defines it for a
  crosstalk budget): the victim driver's supply ``VCC`` (``v_cc``). A deck's
  aggressor is an EMF behind ``R_drive`` into ``C_load`` that settles at its
  full step, a rail-to-rail step, and a push-pull output steps at most
  between ground and its supply (this check's reasoning, not a source
  statement), so VCC bounds the real step from above; ``VOH - VOL``, the
  smallest step the datasheet guarantees, would overstate the budget by
  ``VCC / (VOH - VOL)``. The plan's figures take the swing so (``crosstalk.json``
  P4: 3.3-V LVTTL of SCEA005 Fig. 1, 0.4 V / 3 of 3.3 V = 4.0 %; an ATmega128
  at 3 V, 0.1 V / 3 of 3 V = 1.1 %; its Bogatin entry, 5 % of 3.3 V). The
  plan's equal-swing assumption takes the aggressor's step equal to this
  swing; an aggressor on a higher supply puts proportionally more noise on
  the victim, which no deck models - every result's details say so.

A missing fact is NOT_VERIFIED naming the part and the key; a connector's
levels are the off-board part's (named); a receiver whose grounded facts
state more than one input threshold (a ``v_ih*`` / ``v_il*`` key besides
``v_ih`` / ``v_il``, e.g. a RESET VIH2) has per-pin thresholds no
component-level fact can assign - only a stated budget judges it; levels
that leave no margin are named (TI SCEA005: such an interface should not be
used), never judged.

Results: ``tool`` / ``tool_version`` the engine (as spice.si),
``artifact_hash`` the victim's first deck, ``evidence`` every deck and
rawfile with its hash, ``details`` the decks (:data:`XT_DECK_VERSION`), each
run's peak and its time, the matrices, the budget and what the model leaves
out (:data:`XT_MODEL_OMITS`). Decks and rawfiles go to
``workdir/spice_xtalk/<deck_stem(victim)>-<8 hex of the aggressor's SHA-256>-<span>``
(``-o<k>`` per orientation when there are two or more). Victims are
simulated worst matched-line estimate first, until :data:`MAX_XTALK_DECKS`
decks; the rest are NOT_VERIFIED "not simulated: below the N worst by the
matched-line estimate". A victim that alone needs more than
:data:`MAX_XTALK_DECKS` decks is NOT_VERIFIED saying so, and the victims
after it still run. Every ``spice.xtalk.<net>`` recorded earlier that
this run does not produce gets a superseding NOT_APPLICABLE.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_eda.compilers.spice import analysis_command, build
from ai_eda.errors import CompileError, ToolExecutionError, ToolUnavailableError
from ai_eda.ir import (
    AnalysisSpec,
    CircuitIR,
    Component,
    Evidence,
    Net,
    NetClass,
    NetKind,
    PinRef,
    ProjectMeta,
    Provenance,
    ProvenanceKind,
    SimulationSetup,
    SpiceBinding,
    SpiceDevice,
    Stimulus,
    StimulusKind,
    Traced,
    ValidationResult,
    ValidationStatus,
)
from ai_eda.ir.arc import point_arc_fraction
from ai_eda.tools.calc.tline import C0
from ai_eda.tools.calc.xtalk import maxwell_from_modes
from ai_eda.tools.kicad.library import KicadLibrary
from ai_eda.tools.si.driver import DRIVING_TYPES, driver_value
from ai_eda.tools.si.measure import NetMeasure, line_model, measure_nets
from ai_eda.tools.si.paths import NetPath, _pt_seg_t, net_pads
from ai_eda.tools.spice.runner import SpiceAnalysis, SpiceResult, SpiceRunner
from ai_eda.tools.spice.si_check import CONNECTOR_LIBRARY_PREFIX, HOLD_EDGES, HOLD_TAUS, STEPS_PER_EDGE, _pins, deck_stem, topology_problem

S = ValidationStatus
#: check id prefix (``spice.xtalk.<victim>``; not ``spice.si.*``, which the reflection checks, report and figures read)
CHECK_PREFIX = "spice.xtalk"
#: where the decks and rawfiles go, under the workdir
XT_DIR = "spice_xtalk"
#: the deck layout version (``details["deck_version"]``)
XT_DECK_VERSION = "0.1"
#: the deck version of a board with an arc track (:func:`deck_version`): the same layout, its spans placed and its lines timed along
#: paths that measure an arc at its arc length - a mechanism that acts only on those boards stamps its own; every other board stays 0.1
XT_DECK_ARC_VERSION = "0.2"
#: decks per run (each deck runs at most twice: the self-check), worst matched-line estimate first - a runtime bound, not a threshold:
#: the engine time of one SPICE stage is at most MAX_XTALK_DECKS x 2 x RUN_TIMEOUT_S (16 min)
MAX_XTALK_DECKS = 8
#: transient steps a deck may take (stop / step) - a runtime bound: a finer deck is NOT_VERIFIED "deck too fine"
MAX_POINTS = 200_000
#: seconds one run of one deck may take in the engine (a runtime bound: measured 2026-11-13, a T line at a CPL port can make
#: ngspice-46 crawl at femtosecond steps - module docstring); a run past it is NOT_VERIFIED, never a verdict
RUN_TIMEOUT_S = 60.0
#: the self-check (module docstring; choices taken from the 2026-11-13 CPL measurement's criteria, ngspice_shared item 12d)
STEP_CHECK_FACTOR = 2.0
SETTLE_TOL = 1e-3
AGREE_REL = 0.10
AGREE_ABS = 1e-4
#: positions on a path: a span's end point lies on a path piece within this distance (mm)
SPAN_TOL_MM = 1e-5
#: a pre / post line shorter than this (mm) is no line: the CPL port is the pad
LINE_MIN_MM = 1e-6
#: seconds one engine process may take (the run's own timeout plus loading Python and the DLL)
CHILD_TIMEOUT_S = RUN_TIMEOUT_S + 60.0
#: what the deck leaves out (every result's details)
XT_MODEL_OMITS = [
    "each deck holds one victim and one aggressor; every other line is absent (superposition sums the decks)",
    "a span uses the Maxwell matrices of its smallest gap (exact only with one gap)",
    "the lines before and after a span are lossless T lines at the Z0 of the net's main width (bends, neck-downs and vias lumped)",
    "the CPL is lossless but for R = 1 ohm/m on the diagonal (a numerical choice, ngspice_shared item 12c), no G, no solder mask, no dispersion",
    "pads and vias are not aggressors; power and ground copper beside the victim is not modelled",
    "the aggressor's step is taken equal to the victim's swing (equal swing; an NM / 3 budget's swing is the victim driver's supply v_cc), "
    "its edge a 0-100 % linear ramp of t_r: an aggressor on a higher supply puts proportionally more noise on the victim",
]
SELF_CHECK = (f"each deck runs at its step and at {STEP_CHECK_FACTOR:g} x it, each in a process of its own; its peak is used only when both runs succeed, "
              f"settle within {SETTLE_TOL:g} of the step at their DC levels and agree within {AGREE_REL:.0%} + {AGREE_ABS:g} (a choice of this check: "
              "ngspice-46's CPL can simulate an unmatched pair wrongly with no message, ngspice_shared item 12d)")
#: why an analog / rf victim needs a stated budget
NO_ANALOG_BUDGET = "no opened source gives a numeric crosstalk budget for an analog or rf victim, so only a budget the user states judges it"
#: why a coupled piece that runs over an arc piece of a path gets no deck (:func:`plan_victim`)
CURVED_SPAN = "the coupled-line deck models a straight span only, curved copper is not computed"
_PER_PIN_THRESHOLD = re.compile(r"^v_i[hl].+$")

_PROV = Provenance(kind=ProvenanceKind.DERIVED, tool="spice.xtalk", tool_version=XT_DECK_VERSION, note="spice.xtalk deck element (see ai_eda.tools.spice.xtalk_check)")


# --------------------------------------------------------------------------- the plan: spans on both nets' paths


@dataclass(frozen=True)
class XtPiece:
    """One computed parallel piece of a victim's ``si.crosstalk`` row, with the overlap's end points on both tracks and its modal values."""

    aggressor: str
    layer: str
    width_mm: float
    gap_mm: float
    length_mm: float
    victim_span: tuple[tuple[float, float], tuple[float, float]]
    aggressor_span: tuple[tuple[float, float], tuple[float, float]]
    z_even: float
    z_odd: float
    e_eff_even: float
    e_eff_odd: float


@dataclass(frozen=True)
class Span:
    """Consecutive pieces of one aggressor along the victim's path (module docstring), placed on both paths (mm from ``ends[0]``).

    ``v_pos`` is ``(p0, p1)`` on the victim's path, ``p0 < p1``; ``a_pos`` the
    aggressor's positions beside ``p0`` and beside ``p1`` (either order).
    ``piece`` is the piece of the smallest gap (its modal values make the
    matrices); ``exact`` when every piece has that gap.
    """

    aggressor: str
    index: int
    layer: str
    width_mm: float
    gaps: tuple[float, ...]
    v_pos: tuple[float, float]
    a_pos: tuple[float, float]
    piece: XtPiece

    @property
    def gap_mm(self) -> float:
        return min(self.gaps)

    @property
    def exact(self) -> bool:
        return max(self.gaps) - min(self.gaps) <= 1e-6

    @property
    def length_mm(self) -> float:
        return self.v_pos[1] - self.v_pos[0]


@dataclass(frozen=True)
class NetLine:
    """A deck net: its extracted path, the Z0 of its main width, and the path ends (0 / 1) its driver may sit at (one end, or both)."""

    net: str
    path: NetPath
    z0_ohm: float
    drive_ends: tuple[int, ...]


@dataclass(frozen=True)
class VictimPlan:
    """What :func:`plan_victim` found buildable: the victim's line, each aggressor's line, the spans."""

    victim: NetLine
    aggressors: dict[str, NetLine]
    spans: tuple[Span, ...]

    @property
    def decks(self) -> int:
        return sum(len(self.victim.drive_ends) * len(self.aggressors[s.aggressor].drive_ends) for s in self.spans)


def deck_version(ir: CircuitIR) -> str:
    """The deck version ``ir``'s ``spice.xtalk`` results carry (``details["deck_version"]``): :data:`XT_DECK_ARC_VERSION` when its board
    holds an arc track (``Track.mid``: the paths the spans are placed on measure it at its arc length), else :data:`XT_DECK_VERSION` -
    a board without one is planned and stamped exactly as before."""
    pcb = ir.pcb
    return XT_DECK_ARC_VERSION if pcb is not None and any(getattr(t, "mid", None) is not None for t in pcb.tracks) else XT_DECK_VERSION


def _locate(path: NetPath, layer: str, pt: tuple[float, float]) -> float | None:
    """The position (mm from ``path.ends[0]``) of ``pt`` on a piece of ``path`` on ``layer``, or ``None`` when it lies on none.

    On a piece of an arc (``PathEdge.mid``) the point is measured against the arc itself: its exact distance from the arc, and the
    fraction of the piece's length at the nearest point - never against the chord between the piece's ends.
    """
    pos = 0.0
    for e in path.edges:
        if e.layer == layer:
            d, t = _pt_seg_t(pt[0], pt[1], e.start, e.end) if e.mid is None else point_arc_fraction(pt, e.start, e.mid, e.end)
            if d <= SPAN_TOL_MM:
                return pos + t * e.length_mm
        pos += e.length_mm
    return None


def _on_arc(path: NetPath, a: float, b: float) -> bool:
    """Whether ``path`` between the positions ``a`` and ``b`` (mm, either order) runs over a piece of an arc for more than :data:`SPAN_TOL_MM`."""
    lo, hi = min(a, b), max(a, b)
    pos = 0.0
    for e in path.edges:
        if e.mid is not None and min(hi, pos + e.length_mm) - max(lo, pos) > SPAN_TOL_MM:
            return True
        pos += e.length_mm
    return False


def path_delay(path: NetPath, a: float, b: float) -> float | None:
    """The delay (s) of ``path`` between positions ``a`` and ``b`` (mm): each piece's delay prorated; ``None`` when a piece has none."""
    pos = 0.0
    total = 0.0
    for e in path.edges:
        lo, hi = max(a, pos), min(b, pos + e.length_mm)
        if hi > lo and e.length_mm > 0.0:
            if e.delay_s is None:
                return None
            total += e.delay_s * (hi - lo) / e.length_mm
        pos += e.length_mm
    return total


def _pad_drives(ir: CircuitIR, label: str) -> bool:
    ref, _, number = label.partition(".")
    comp = ir.component(ref)
    pin = comp.pin(number) if comp is not None else None
    return pin is not None and pin.electrical_type in DRIVING_TYPES


def _net_line(ir: CircuitIR, net: str, measures: dict[str, NetMeasure], role: str) -> NetLine | str:
    si = ir.si
    assert si is not None
    cls = si.class_of(net)
    topo = topology_problem(ir, net, None if cls is None else cls.driver)
    if topo is not None:
        return f"{role} {topo}"
    m = measures.get(net)
    path = m.path if m is not None else None
    if path is None:
        why = m.path_problem if m is not None and m.path_problem else "not measured"
        return f"{role} {net} has no extracted pad-to-pad path ({why})"
    if not path.edges:
        return f"{role} {net}: its path has no pieces"
    if path.bound or path.delay_s is None:
        return f"{role} {net}: its line has no delay over a plane (impedance is undefined without a reference plane)"
    stackup = ir.pcb.stackup if ir.pcb is not None else None
    model, why = line_model(stackup, *path.main) if path.main is not None else (None, "no track on the path")
    if model is None or model.z0_ohm is None:
        return f"{role} {net}: no Z0 at its main width ({why or (model.reason if model is not None else '')})"
    drivers = tuple(i for i, label in enumerate(path.ends) if _pad_drives(ir, label))
    return NetLine(net=net, path=path, z0_ohm=float(model.z0_ohm), drive_ends=drivers if len(drivers) == 1 else (0, 1))


def plan_victim(ir: CircuitIR, victim: str, pieces: list[XtPiece], measures: dict[str, NetMeasure]) -> VictimPlan | str:
    """The decks of ``victim`` (module docstring), or why they cannot be built: more than two pads, an unmodelled part, no extracted path
    (inner-layer copper, a typed via, no KiCad library ...), a piece off a path, a piece over an arc piece of a path (:data:`CURVED_SPAN`),
    no Z0 - for the victim and for each aggressor."""
    if not pieces:
        return "no computed parallel piece"
    v = _net_line(ir, victim, measures, "victim")
    if isinstance(v, str):
        return v
    lines: dict[str, NetLine] = {}
    placed: dict[str, list[tuple[float, float, float, float, XtPiece]]] = {}
    for p in pieces:
        if p.aggressor not in lines:
            a = _net_line(ir, p.aggressor, measures, "aggressor")
            if isinstance(a, str):
                return a
            lines[p.aggressor] = a
        a = lines[p.aggressor]
        v0, v1 = _locate(v.path, p.layer, p.victim_span[0]), _locate(v.path, p.layer, p.victim_span[1])
        a0, a1 = _locate(a.path, p.layer, p.aggressor_span[0]), _locate(a.path, p.layer, p.aggressor_span[1])
        if v0 is None or v1 is None:
            return f"a coupled piece beside {p.aggressor} on {p.layer} lies on copper of {victim} off its pad-to-pad path {v.path.label}"
        if a0 is None or a1 is None:
            return f"a coupled piece of aggressor {p.aggressor} on {p.layer} lies on its copper off its pad-to-pad path {a.path.label}"
        if v0 > v1:
            v0, v1, a0, a1 = v1, v0, a1, a0
        for line, lo, hi in ((v, v0, v1), (a, a0, a1)):
            if _on_arc(line.path, lo, hi):
                return f"a coupled piece beside {p.aggressor} on {p.layer} runs over an arc track of {line.net} (its path {line.path.label}): {CURVED_SPAN}"
        if abs((v1 - v0) - p.length_mm) > 1e-4 or abs(abs(a1 - a0) - p.length_mm) > 1e-4:
            return f"a coupled piece beside {p.aggressor} on {p.layer} is not one straight stretch of both paths"
        placed.setdefault(p.aggressor, []).append((v0, v1, a0, a1, p))
    spans: list[Span] = []
    for aggr, items in placed.items():
        items.sort(key=lambda x: (x[0], x[1]))
        groups: list[list[tuple[float, float, float, float, XtPiece]]] = []
        for it in items:
            last = groups[-1][-1] if groups else None
            if (last is not None and abs(it[0] - last[1]) <= SPAN_TOL_MM and abs(it[2] - last[3]) <= SPAN_TOL_MM and it[4].layer == last[4].layer
                    and abs(it[4].width_mm - last[4].width_mm) <= 1e-6):
                groups[-1].append(it)
            else:
                groups.append([it])
        for k, g in enumerate(groups):
            smallest = min((x[4] for x in g), key=lambda x: x.gap_mm)
            spans.append(Span(aggressor=aggr, index=k, layer=g[0][4].layer, width_mm=g[0][4].width_mm, gaps=tuple(x[4].gap_mm for x in g),
                              v_pos=(g[0][0], g[-1][1]), a_pos=(g[0][2], g[-1][3]), piece=smallest))
    return VictimPlan(victim=v, aggressors=lines, spans=tuple(spans))


# --------------------------------------------------------------------------- the deck


def _traced(value: float, unit: str | None, note: str) -> Traced:
    return Traced(value=value, unit=unit, provenance=_PROV.model_copy(update={"note": note}))


@dataclass
class XtDeck:
    """One deck: its IR, transient, the node judged (the victim's receiver pad), the aggressor's far pad, and what went into it."""

    ir: CircuitIR
    step: float
    stop: float
    hold: float
    victim_node: str
    aggressor_far: str
    info: dict[str, Any]


def xtalk_deck(project_id: str, *, victim: NetLine, aggressor: NetLine, span: Span, v_drive: int, a_drive: int, t_r: float,
               r_v: float, c_v: float, r_a: float, c_a: float) -> XtDeck | str:
    """The deck (module docstring) of ``span`` with the victim driven at path end ``v_drive`` and the aggressor at ``a_drive``, or why not."""
    vp, ap = victim.path, aggressor.path
    p0, p1 = span.v_pos
    a_in, a_out = span.a_pos
    lv, la = vp.length_mm, ap.length_mm
    pc = span.piece
    mx = maxwell_from_modes(pc.z_even, pc.z_odd, pc.e_eff_even, pc.e_eff_odd)
    length_m = (p1 - p0) / 1000.0
    td_slow = length_m * math.sqrt(max(pc.e_eff_even, pc.e_eff_odd)) / C0
    td_fast = length_m * math.sqrt(min(pc.e_eff_even, pc.e_eff_odd)) / C0
    # the four pre / post lines: (name, from node, to node, path, a, b, z0)
    lines: list[tuple[str, str, str, NetPath, float, float, float]] = []
    vi = "VI" if p0 > LINE_MIN_MM else "V0"
    vo = "VO" if lv - p1 > LINE_MIN_MM else "V1"
    if vi != "V0":
        lines.append(("TV0", "V0", "VI", vp, 0.0, p0, victim.z0_ohm))
    if vo != "V1":
        lines.append(("TV1", "VO", "V1", vp, p1, lv, victim.z0_ohm))
    lo_port, hi_port = ("AI", "AO") if a_in <= a_out else ("AO", "AI")
    lo, hi = min(a_in, a_out), max(a_in, a_out)
    names = {lo_port: lo_port if lo > LINE_MIN_MM else "A0", hi_port: hi_port if la - hi > LINE_MIN_MM else "A1"}
    if names[lo_port] != "A0":
        lines.append(("TA0", "A0", lo_port, ap, 0.0, lo, aggressor.z0_ohm))
    if names[hi_port] != "A1":
        lines.append(("TA1", hi_port, "A1", ap, hi, la, aggressor.z0_ohm))
    ai, ao = names["AI"], names["AO"]
    tds: dict[str, float] = {}
    for name, _, _, path, a, b, _ in lines:
        td = path_delay(path, a, b)
        if td is None or td <= 0.0:
            return f"no delay for the {name} piece of {path.label} ({a:.6g}..{b:.6g} mm)"
        tds[name] = td
    v_total = sum(td for n, td in tds.items() if n.startswith("TV")) + td_slow
    a_total = sum(td for n, td in tds.items() if n.startswith("TA")) + td_slow
    hold = HOLD_EDGES * (t_r + 2.0 * max(v_total, a_total)) + HOLD_TAUS * max((r_v + victim.z0_ohm) * c_v, (r_a + aggressor.z0_ohm) * c_a)
    step = min(t_r, td_fast, *tds.values()) / STEPS_PER_EDGE
    stop = t_r + hold
    v_pads, a_pads = ("V0", "V1"), ("A0", "A1")
    v_drv, v_rx, a_drv, a_far = v_pads[v_drive], v_pads[1 - v_drive], a_pads[a_drive], a_pads[1 - a_drive]
    ir = CircuitIR(project=ProjectMeta(id=project_id, name=project_id))

    def part(ref: str, n: int, binding: SpiceBinding) -> Component:
        return Component(ref=ref, value=ref, pins=_pins(n), provenance=_PROV, spice=binding)

    comps = [
        part("RSV", 2, SpiceBinding(device=SpiceDevice.R, value=_traced(r_v, "ohm", "victim driver source resistance (holds a level)"), provenance=_PROV)),
        part("CLV", 2, SpiceBinding(device=SpiceDevice.C, value=_traced(c_v, "F", "victim receiver load"), provenance=_PROV)),
        part("RSA", 2, SpiceBinding(device=SpiceDevice.R, value=_traced(r_a, "ohm", "aggressor driver source resistance"), provenance=_PROV)),
        part("CLA", 2, SpiceBinding(device=SpiceDevice.C, value=_traced(c_a, "F", "aggressor far-end load"), provenance=_PROV)),
        part("P1", 6, SpiceBinding(device=SpiceDevice.P, pin_order=["1", "2", "3", "4", "5", "6"], params={
            "l11": _traced(mx.l11, "H/m", "maxwell_from_modes L11 at the span's smallest gap"), "l12": _traced(mx.l12, "H/m", "maxwell_from_modes L12"),
            "l22": _traced(mx.l11, "H/m", "L22 = L11 (equal widths)"), "c11": _traced(mx.c11, "F/m", "maxwell_from_modes C11"),
            "c22": _traced(mx.c11, "F/m", "C22 = C11 (equal widths)"), "cm": _traced(-mx.c12, "F/m", "cm = -C12 (Maxwell mutual term)"),
            "length": _traced(length_m, "m", "the span's length along the victim's path")}, provenance=_PROV)),
    ]
    for name, _, _, _, _, _, z0 in lines:
        comps.append(part(name, 4, SpiceBinding(device=SpiceDevice.T, pin_order=["1", "2", "3", "4"], params={
            "z0": _traced(z0, "ohm", "calc.tline.microstrip.z0 of the net's main width"), "td": _traced(tds[name], "s", "the path pieces' summed delay")},
            provenance=_PROV)))
    ir.components = comps
    pins: dict[str, list[tuple[str, str]]] = {"S": [("RSA", "1")]}

    def on(node: str, ref: str, pin: str) -> None:
        pins.setdefault(node, []).append((ref, pin))

    on(v_drv, "RSV", "1")
    on(v_rx, "CLV", "1")
    on(a_drv, "RSA", "2")
    on(a_far, "CLA", "1")
    for pin, node in zip(("1", "2", "4", "5"), (vi, ai, vo, ao)):
        on(node, "P1", pin)
    for name, a, b, _, _, _, _ in lines:
        on(a, name, "1")
        on(b, name, "3")
    ground = [("RSV", "2"), ("CLV", "2"), ("CLA", "2"), ("P1", "3"), ("P1", "6")]
    for name, *_ in lines:
        ground += [(name, "2"), (name, "4")]
    ir.nets = [Net(name=n, kind=NetKind.SIGNAL, pins=[PinRef(component_ref=r, pin_number=p) for r, p in ps], provenance=_PROV) for n, ps in pins.items()]
    ir.nets.append(Net(name="GND", kind=NetKind.GROUND, pins=[PinRef(component_ref=r, pin_number=p) for r, p in ground], provenance=_PROV))
    pulse = {"v1": 0.0, "v2": 1.0, "td": 0.0, "tr": t_r, "tf": t_r, "pw": stop, "per": 2.0 * (t_r + stop)}
    units = {"v1": "V", "v2": "V", "td": "s", "tr": "s", "tf": "s", "pw": "s", "per": "s"}
    ir.simulation = SimulationSetup(
        stimuli=[Stimulus(id="VDRV", source="voltage", net="S", reference_net="GND", kind=StimulusKind.PULSE,
                          params={k: _traced(val, units[k], f"pulse {k} (held: the step never falls inside the run)") for k, val in pulse.items()},
                          provenance=_PROV)],
        analyses=[AnalysisSpec(id="tran", kind=SpiceAnalysis.TRAN, params={"step": _traced(step, "s", "tran step"), "stop": _traced(stop, "s", "tran stop")},
                               provenance=_PROV)],
    )
    info = {"victim_driver": vp.ends[v_drive], "victim_receiver": vp.ends[1 - v_drive], "aggressor_driver": ap.ends[a_drive],
            "aggressor_far": ap.ends[1 - a_drive], "victim_path": vp.label, "aggressor_path": ap.label,
            "victim_pos_mm": [round(p0, 6), round(p1, 6)], "aggressor_pos_mm": [round(a_in, 6), round(a_out, 6)],
            "lines": {name: {"from": a, "to": b, "length_mm": round(e - s, 6), "z0_ohm": round(z0, 4), "td_s": tds[name]}
                      for name, a, b, _, s, e, z0 in lines},
            "cpl": {"length_m": length_m, "l11": mx.l11, "l12": mx.l12, "c11": mx.c11, "c12": mx.c12, "z_even_ohm": pc.z_even, "z_odd_ohm": pc.z_odd,
                    "e_eff_even": pc.e_eff_even, "e_eff_odd": pc.e_eff_odd, "td_even_s": td_slow if pc.e_eff_even >= pc.e_eff_odd else td_fast,
                    "td_odd_s": td_fast if pc.e_eff_even >= pc.e_eff_odd else td_slow},
            "t_r_s": t_r, "r_drive_victim_ohm": r_v, "c_load_victim_f": c_v, "r_drive_aggressor_ohm": r_a, "c_load_aggressor_f": c_a,
            "step_s": step, "stop_s": stop, "hold_s": hold, "points": int(math.ceil(stop / step))}
    return XtDeck(ir=ir, step=step, stop=stop, hold=hold, victim_node=v_rx.lower(), aggressor_far=a_far.lower(), info=info)


# --------------------------------------------------------------------------- the budget


@dataclass(frozen=True)
class Budget:
    """A victim's budget (fraction of the swing) and where it came from, or ``None`` with what is missing."""

    value: float | None
    source: str
    missing: str = ""
    details: dict[str, Any] = field(default_factory=dict)


def _fact(ir: CircuitIR, ref: str, key: str, unit: str) -> tuple[float | None, str]:
    comp = ir.component(ref)
    fact = comp.electrical.get(key) if comp is not None else None
    if (fact is None or not fact.provenance.is_authoritative or fact.unit != unit or isinstance(fact.value, bool)
            or not isinstance(fact.value, (int, float)) or not math.isfinite(float(fact.value))):
        return None, f"{ref}.{key}"
    return float(fact.value), f"{ref}.{key} (grounded datasheet fact)"


def _is_connector(ir: CircuitIR, ref: str) -> bool:
    comp = ir.component(ref)
    return comp is not None and comp.symbol is not None and comp.symbol.library.startswith(CONNECTOR_LIBRARY_PREFIX)


def nm3_budget(ir: CircuitIR, driver: str, receiver: str) -> tuple[float | None, dict[str, Any], str]:
    """``(budget, details, "")`` = min(VOH - VIH, VIL - VOL) / 3 / VCC of driver part ``driver`` into receiver part ``receiver``
    (module docstring: VCC is the swing), or ``(None, details, why)`` naming each missing part and key."""
    det: dict[str, Any] = {"driver": driver, "receiver": receiver}
    off = [f"{r} is a connector: the levels are the off-board part's behind it" for r in dict.fromkeys((driver, receiver)) if _is_connector(ir, r)]
    if off:
        return None, det, "; ".join(off)
    rx = ir.component(receiver)
    extra = sorted(k for k in (rx.electrical if rx is not None else {}) if _PER_PIN_THRESHOLD.match(k) and k not in ("v_ih", "v_il"))
    if extra:
        return None, det, (f"{receiver}'s grounded facts state more than one input threshold ({', '.join(extra)}): which pin takes which is not a "
                           "component-level fact (a per-pin threshold such as a RESET VIH2), so only a stated budget judges it")
    values: dict[str, float] = {}
    sources: dict[str, str] = {}
    missing: list[str] = []
    for ref, key in ((driver, "v_oh"), (driver, "v_ol"), (driver, "v_cc"), (receiver, "v_ih"), (receiver, "v_il")):
        v, src = _fact(ir, ref, key, "V")
        if v is None:
            missing.append(src)
        else:
            values[key], sources[key] = v, src
    for key in ("i_oh", "i_ol"):
        v, src = _fact(ir, driver, key, "A")
        det[key] = v
        if v is not None:
            sources[key] = src
    det.update(values)
    det["sources"] = sources
    if missing:
        return None, det, "missing the grounded datasheet fact(s) " + ", ".join(missing)
    nmh, nml = values["v_oh"] - values["v_ih"], values["v_il"] - values["v_ol"]
    swing = values["v_cc"]
    det.update(nmh_v=nmh, nml_v=nml, swing_v=swing,
               row="v_oh at i_oh, v_ol at i_ol, at v_cc (the grounded rows)" if det["i_oh"] is not None and det["i_ol"] is not None
               else "the row current of v_oh / v_ol was not grounded (i_oh / i_ol): the levels are read as grounded")
    if values["v_oh"] <= values["v_ol"]:
        return None, det, f"{driver}'s v_oh {values['v_oh']:g} V is not above its v_ol {values['v_ol']:g} V: no swing"
    if swing <= 0.0 or swing < values["v_oh"]:
        return None, det, (f"{driver}'s v_cc {swing:g} V is not a supply at or above its v_oh {values['v_oh']:g} V: it is not the supply "
                           "the levels were read at, so it bounds no step")
    if min(nmh, nml) <= 0.0:
        return None, det, (f"the levels leave no noise margin (NMH = {nmh:.6g} V, NML = {nml:.6g} V; TI SCEA005: such an interface should not be used): "
                           "a level problem, crosstalk is not judged against it")
    budget = min(nmh, nml) / 3.0 / swing
    det["budget_rel"] = budget
    return budget, det, ""


def victim_budget(ir: CircuitIR, cls: NetClass | None, victim: str, kind: NetKind, directions: list[tuple[str, str]]) -> Budget:
    """The budget of ``victim`` (module docstring): the class's stated ``xtalk_budget_rel`` first, else NM / 3 over ``directions``
    ((driver part, receiver part) pairs: the smallest), else ``None`` naming what is missing."""
    if cls is None:
        return Budget(None, "", f"{victim} is in no net class (no default class): no budget is stated")
    key = f"si.net_classes[{cls.name}].xtalk_budget_rel"
    if cls.xtalk_budget_rel is not None:
        return Budget(float(cls.xtalk_budget_rel.value), f"{key} (stated by the user)", details={"kind": "stated", "value": float(cls.xtalk_budget_rel.value)})
    if kind in (NetKind.ANALOG, NetKind.RF):
        return Budget(None, "", f"{key}: {NO_ANALOG_BUDGET}")
    if not directions:
        return Budget(None, "", f"{key}, or the grounded levels of a pad that drives {victim}: no pad of {victim} can drive it")
    rows: list[dict[str, Any]] = []
    whys: list[str] = []
    values: list[float] = []
    for drv, rx in directions:
        v, det, why = nm3_budget(ir, drv, rx)
        rows.append(det)
        if v is None:
            whys.append(f"{drv} -> {rx}: {why}")
        else:
            values.append(v)
    details = {"kind": "NM/3", "rule": ("min(VOH_drv - VIH_rx, VIL_rx - VOL_drv) / 3 / swing, swing = VCC_drv (the driver's grounded v_cc: an upper "
                                        "bound of its step, a choice; TI SCEA005 Fig. 1; Bogatin RoT #19)"),
               "directions": rows}
    if whys:
        return Budget(None, "", f"{key} (a stated budget), or NM / 3 from grounded datasheet levels: " + "; ".join(whys), details)
    value = min(values)
    return Budget(value, "NM / 3 of the grounded datasheet levels (" + "; ".join(f"{d} -> {r}" for d, r in directions) + ")", details={**details, "value": value})


def _directions(ir: CircuitIR, line: NetLine) -> list[tuple[str, str]]:
    """``(driver part, receiver part)`` of each orientation whose driver end can drive: a pin of ``DRIVING_TYPES`` or a connector's pad."""
    out: list[tuple[str, str]] = []
    for d in line.drive_ends:
        drv, rx = line.path.ends[d], line.path.ends[1 - d]
        ref = drv.partition(".")[0]
        if _pad_drives(ir, drv) or _is_connector(ir, ref):
            out.append((ref, rx.partition(".")[0]))
    return out


# --------------------------------------------------------------------------- running the decks


DeckRunner = Callable[[Path, Path, str, tuple[str, ...]], SpiceResult]

_CHILD = r"""
import json, sys
from pathlib import Path
from ai_eda.tools.spice.ngspice_shared import NgspiceShared
from ai_eda.tools.spice.runner import SpiceAnalysis
a = json.loads(sys.argv[1])
runner = NgspiceShared(a["dll"], a["codemodels"]) if a["dll"] else NgspiceShared()
res = runner.run(Path(a["deck"]), SpiceAnalysis.TRAN, Path(a["out"]), a["command"], timeout_s=a["timeout"])
keep = set(a["keep"])
res = res.model_copy(update={"vectors": {k: v for k, v in res.vectors.items() if k in keep}})
Path(a["result"]).write_text(res.model_dump_json(), encoding="utf-8")
"""


def _child_runner(runner: SpiceRunner) -> DeckRunner:
    """Each deck in a Python process of its own (ngspice_shared item 12f: a later CPL deck can crash the process that loaded one)."""
    import ai_eda

    root = str(Path(ai_eda.__file__).resolve().parent.parent)
    dll = getattr(runner, "dll_path", None)
    models = getattr(runner, "codemodel_dir", None)

    def run(deck: Path, out: Path, command: str, keep: tuple[str, ...]) -> SpiceResult:
        out.mkdir(parents=True, exist_ok=True)
        result = out / "result.json"
        if result.exists():
            result.unlink()
        args = {"dll": None if dll is None else str(dll), "codemodels": None if models is None else str(models), "deck": str(deck), "out": str(out),
                "command": command, "keep": list(keep), "result": str(result), "timeout": RUN_TIMEOUT_S}
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(x for x in (root, os.environ.get("PYTHONPATH", "")) if x)}
        base = dict(engine=runner.engine, engine_version="unknown", netlist_path=str(deck), netlist_hash=SpiceRunner.netlist_hash(deck),
                    analysis=SpiceAnalysis.TRAN, command=command, succeeded=False)
        try:
            proc = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(args)], cwd=out, env=env, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=CHILD_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return SpiceResult(**base, timed_out=True, unverifiable=f"not run: the engine process took longer than {CHILD_TIMEOUT_S:g} s")
        if proc.returncode != 0 or not result.is_file():
            tail = " ".join((proc.stderr or "").split())[-400:]
            return SpiceResult(**base, unverifiable=f"not run: the engine process exited {proc.returncode} ({tail or 'no output'})")
        return SpiceResult.model_validate_json(result.read_text(encoding="utf-8"))

    return run


def deck_runner(runner: SpiceRunner) -> DeckRunner:
    """How each deck runs: KiCad's ngspice.dll in a process of its own per deck; any other runner (the batch binary already runs a process
    per deck; a test's stub) directly."""
    from ai_eda.tools.spice.ngspice_shared import NgspiceShared

    if isinstance(runner, NgspiceShared):
        return _child_runner(runner)
    return lambda deck, out, command, keep: runner.run(deck, SpiceAnalysis.TRAN, out, command)


def _peak(res: SpiceResult, node: str) -> tuple[float, float] | str:
    """``(peak |v|, its time)`` of ``node``, or why not."""
    try:
        time, v = res.scale_values(), res.vector(node)
    except (KeyError, ValueError) as e:
        return f"vector not produced: {e}"
    if not v or len(v) != len(time) or any(not math.isfinite(x) for x in (*time, *v)):
        return "the waveform is empty or not finite"
    k = max(range(len(v)), key=lambda i: abs(v[i]))
    return abs(v[k]), time[k]


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class DeckRun:
    """One deck's two runs and its verdict on itself (module docstring)."""

    span: Span
    v_drive: int
    a_drive: int
    path: Path
    deck_hash: str
    text: str
    info: dict[str, Any]
    peak: float | None = None
    problem: str | None = None
    runs: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    engine_version: str | None = None

    @property
    def confirmed(self) -> bool:
        return self.peak is not None and self.problem is None


def run_deck(run: DeckRunner, deck: XtDeck, dr: DeckRun, commands: list[str]) -> None:
    """Both runs of one deck and the self-check (module docstring); fills ``dr``."""
    keep = ("time", deck.victim_node, deck.aggressor_far)
    peaks: list[float] = []
    for k, command in enumerate(commands):
        out = dr.path.parent / f"{dr.path.stem}.r{k + 1}"
        try:
            res = run(dr.path, out, command, keep)
        except (ToolUnavailableError, ToolExecutionError) as e:
            dr.problem = f"not run: {e}"
            return
        dr.engine_version = dr.engine_version or (res.engine_version if res.engine_version != "unknown" else None)
        row: dict[str, Any] = {"command": res.command, "succeeded": res.succeeded, "n_points": res.n_points}
        dr.runs.append(row)
        if res.raw_output_path:
            dr.evidence.append(Evidence(description=f"ngspice rawfile of {dr.path.stem} ({res.command})", path=res.raw_output_path, content_hash=res.raw_output_hash))
        if not res.succeeded:
            why = res.unverifiable or "; ".join(res.errors[:3]) or "the run did not succeed"
            row["errors"] = list(res.errors[:5])
            dr.problem = f"run {k + 1} ({res.command}) did not succeed: {why}"
            return
        pk = _peak(res, deck.victim_node)
        if isinstance(pk, str):
            dr.problem = f"run {k + 1}: {pk}"
            return
        try:
            far = res.vector(deck.aggressor_far)
        except KeyError:
            far = []
        if not far or not math.isfinite(far[-1]):
            dr.problem = f"run {k + 1}: the aggressor's far-end vector {deck.aggressor_far!r} was not produced"
            return
        end_v = res.vector(deck.victim_node)[-1]
        row.update(peak_rel=pk[0], t_peak_s=pk[1], final_victim_v=end_v, final_aggressor_far_v=far[-1])
        peaks.append(pk[0])
        if abs(end_v) > SETTLE_TOL or abs(far[-1] - 1.0) > SETTLE_TOL:
            dr.problem = (f"run {k + 1} ({res.command}) did not settle at its DC levels (victim {end_v:.6g} V of 0, aggressor far end "
                          f"{far[-1]:.6g} V of 1; tolerance {SETTLE_TOL:g}): ngspice's CPL can simulate an unmatched pair wrongly with no message "
                          "(ngspice_shared item 12d)")
            return
    if len(peaks) == 2 and abs(peaks[0] - peaks[1]) > AGREE_REL * max(peaks) + AGREE_ABS:
        dr.problem = (f"the two steps disagree: peak {peaks[0]:.6g} at the step, {peaks[1]:.6g} at {STEP_CHECK_FACTOR:g} x it "
                      f"(more than {AGREE_REL:.0%} + {AGREE_ABS:g}): ngspice's CPL result depends on the step here (ngspice_shared item 12d)")
        return
    dr.peak = max(peaks)


# --------------------------------------------------------------------------- the check


def _span_label(s: Span) -> str:
    return (f"{s.aggressor} on {s.layer}, span {s.index} ({s.v_pos[0]:.3f}..{s.v_pos[1]:.3f} mm along the victim's path, {s.length_mm:.3f} mm at gap "
            f"{s.gap_mm:.4g} mm" + ("" if s.exact else f", lumped at the smallest of gaps {', '.join(f'{g:.4g}' for g in sorted(set(s.gaps)))}") + ")")


def _judge(victim: str, budget: float, decks: list[DeckRun]) -> tuple[S, str, dict[str, Any]]:
    """The victim's verdict from its decks (module docstring)."""
    groups: dict[tuple[str, int], list[DeckRun]] = {}
    for d in decks:
        groups.setdefault((d.span.aggressor, d.span.index), []).append(d)
    spans: list[dict[str, Any]] = []
    total = 0.0
    unconfirmed = [d for d in decks if not d.confirmed]
    for (_, _), ds in groups.items():
        s = ds[0].span
        ok = all(d.confirmed for d in ds)
        worst = max((d.peak for d in ds if d.peak is not None), default=None)
        best = min((d.peak for d in ds), default=None) if ok else None  # type: ignore[type-var]
        spans.append({"aggressor": s.aggressor, "span": s.index, "layer": s.layer, "gap_mm": s.gap_mm, "exact": s.exact, "length_mm": round(s.length_mm, 6),
                      "worst_peak_rel": worst, "best_orientation_peak_rel": best, "confirmed": ok, "label": _span_label(s)})
        if worst is not None:
            total += worst
    fails = [x for x in spans if x["confirmed"] and x["exact"] and x["best_orientation_peak_rel"] is not None and x["best_orientation_peak_rel"] > budget]
    measured = {"total_rel": total, "budget_rel": budget, "spans": spans}
    if fails:
        worst = max(fails, key=lambda x: x["best_orientation_peak_rel"])
        return S.FAIL, (f"{victim}: the deck of {worst['label']} alone peaks at {worst['best_orientation_peak_rel']:.3%} of the aggressor step "
                        f"> the budget {budget:.3%}"), {**measured, "failing": [x["label"] for x in fails]}
    if unconfirmed:
        why = "; ".join(f"{d.path.stem}: {d.problem}" for d in unconfirmed[:3]) + (" ..." if len(unconfirmed) > 3 else "")
        return S.NOT_VERIFIED, f"{victim}: {len(unconfirmed)} of {len(decks)} deck(s) gave no usable peak - {why}", measured
    if total <= budget:
        return S.PASS, (f"{victim}: the sum of the {len(spans)} span(s)' worst peaks is {total:.3%} of the aggressor step <= the budget {budget:.3%} "
                        "(every deck ran and confirmed itself)"), measured
    whys: list[str] = []
    # an exact span over the budget that is no FAIL: its most favourable orientation stays within it
    for x, ds in zip(spans, groups.values()):
        if x["exact"] and x["worst_peak_rel"] is not None and x["worst_peak_rel"] > budget:
            over = "; ".join(f"victim driven from {d.info['victim_driver']}, aggressor from {d.info['aggressor_driver']}: {d.peak:.3%}"
                             for d in ds if d.peak is not None and d.peak > budget)
            whys.append(f"the deck of {x['label']} exceeds the budget {budget:.3%} only with {over} ({x['best_orientation_peak_rel']:.3%} in its most "
                        "favourable orientation): which end drives cannot be told, so no orientation makes a verdict")
    lumped = [x for x in spans if not x["exact"] and x["worst_peak_rel"] is not None and x["worst_peak_rel"] > budget]
    if lumped:
        whys.append(f"{'' if whys else 'only '}a span lumped at its smallest gap exceeds the budget {budget:.3%} ({'; '.join(x['label'] for x in lumped)}): "
                    "its smallest gap stands for the whole span, an upper bound")
    if whys:
        return S.NOT_VERIFIED, f"{victim}: " + "; ".join(whys), measured
    return S.NOT_VERIFIED, (f"{victim}: only the sum of the spans' peaks, {total:.3%}, exceeds the budget {budget:.3%}; no span alone does - "
                            "aggressors switching together not simulated"), measured


def spice_xtalk_results(ir: CircuitIR, tools: dict[str, Any], workdir: Path | str) -> list[ValidationResult]:
    """Every ``spice.xtalk.<victim>`` result (module docstring), plus superseding NOT_APPLICABLE ones; ``[]`` without ``ir.si``.

    ``tools["kicad_library"]`` gives the pad boxes the paths are extracted
    with (without it no path exists and no victim is handed off);
    ``tools["spice"]`` is the engine.
    """
    from ai_eda.validation.si import crosstalk_scan

    si = ir.si
    if si is None:
        return []
    library = tools.get("kicad_library")
    lib = library if isinstance(library, KicadLibrary) else None
    measures = measure_nets(ir, library=lib)
    pads = net_pads(ir, lib) if lib is not None and ir.pcb is not None else None
    scan = crosstalk_scan(ir, si, measures, pads)
    rows = {r["net"]: r for r in scan.result.details.get("rows", [])}
    order_ir = {n.name: i for i, n in enumerate(ir.nets)}
    runner = tools.get("spice")
    ir_hash = ir.content_hash()
    out: dict[str, ValidationResult] = {}
    handed = sorted(scan.plans, key=lambda n: (-(rows[n].get("matched_estimate_rel") or 0.0), order_ir.get(n, 0)))
    version = deck_version(ir)
    used = 0
    capped = False
    worst_taken = 0
    run: DeckRunner | None = None
    for victim in handed:
        plan = scan.plans[victim]
        check = f"{CHECK_PREFIX}.{victim}"
        vcls = si.class_of(victim)
        base: dict[str, Any] = {"victim": victim, "class": None if vcls is None else vcls.name, "deck_version": version,
                                "matched_estimate_rel": rows[victim].get("matched_estimate_rel"), "model_omits": list(XT_MODEL_OMITS), "self_check": SELF_CHECK,
                                "swing": ("a deck's 1 V aggressor step stands for the victim's swing: the aggressor's step is taken equal to it (equal swing; "
                                          "an NM / 3 budget's swing is the victim driver's supply v_cc, an upper bound of its step); an aggressor on a "
                                          "higher supply puts proportionally more noise on the victim, which no deck models"),
                                "spans": [_span_label(s) for s in plan.spans]}

        def nv(message: str, **extra: Any) -> ValidationResult:
            return ValidationResult(check_id=check, status=S.NOT_VERIFIED, message=message, ir_hash=ir_hash, details={**base, **extra})

        missing: list[str] = []
        values: dict[str, dict[str, float]] = {}
        sources: dict[str, str] = {}
        for net, fields in ((victim, ("r_drive_ohm", "c_load_f")), *((a, ("t_rise_s", "r_drive_ohm", "c_load_f")) for a in plan.aggressors)):
            cls = si.class_of(net)
            for f in fields:
                if cls is None:
                    missing.append(f"{net} is in no net class ({f})")
                    continue
                v, why = driver_value(ir, cls, f, net)
                if v is None:
                    missing.append(why)
                else:
                    values.setdefault(net, {})[f] = v.value
                    sources[f"{net}.{f}"] = v.source
        base["sources"] = sources
        net_kind = next((n.kind for n in ir.nets if n.name == victim), NetKind.SIGNAL)
        budget = victim_budget(ir, vcls, victim, net_kind, _directions(ir, plan.victim))
        base["budget"] = {"value_rel": budget.value, "source": budget.source, **({"missing": budget.missing} if budget.missing else {}), **budget.details}
        if missing:
            out[victim] = nv(f"{victim}: missing " + ", ".join(dict.fromkeys(missing)))
            continue
        if budget.value is None:
            out[victim] = nv(f"{victim}: no crosstalk budget - {budget.missing}")
            continue
        # the decks: (span, victim orientation, aggressor orientation)
        decks: list[tuple[Span, int, int, XtDeck]] = []
        problem = None
        stem_v = deck_stem(victim)
        for s in plan.spans:
            aline = plan.aggressors[s.aggressor]
            av = values[s.aggressor]
            for vd in plan.victim.drive_ends:
                for ad in aline.drive_ends:
                    d = xtalk_deck(f"xt_{stem_v}", victim=plan.victim, aggressor=aline, span=s, v_drive=vd, a_drive=ad, t_r=av["t_rise_s"],
                                   r_v=values[victim]["r_drive_ohm"], c_v=values[victim]["c_load_f"], r_a=av["r_drive_ohm"], c_a=av["c_load_f"])
                    if isinstance(d, str):
                        problem = d
                        break
                    if d.info["points"] > MAX_POINTS:
                        problem = (f"deck too fine: {_span_label(s)} needs {d.info['points']} steps of {d.step:.4g} s to {d.stop:.4g} s "
                                   f"(more than MAX_POINTS = {MAX_POINTS})")
                        break
                    decks.append((s, vd, ad, d))
                if problem:
                    break
            if problem:
                break
        if problem:
            out[victim] = nv(f"{victim}: {problem}")
            continue
        if len(decks) > MAX_XTALK_DECKS:  # never runs, and does not stop the victims after it
            out[victim] = nv(f"{victim}: not simulated: it alone needs {len(decks)} decks, more than MAX_XTALK_DECKS = {MAX_XTALK_DECKS} per run",
                             decks_needed=len(decks))
            continue
        if capped or used + len(decks) > MAX_XTALK_DECKS:  # used > 0 here, so worst_taken > 0
            capped = True
            out[victim] = nv(f"{victim}: not simulated: below the {worst_taken} worst by the matched-line estimate "
                             f"(MAX_XTALK_DECKS = {MAX_XTALK_DECKS} decks per run, {len(decks)} needed here)", decks_needed=len(decks))
            continue
        used += len(decks)
        worst_taken += 1
        if not isinstance(runner, SpiceRunner) or not runner.available():
            out[victim] = nv(f"{victim}: no SPICE engine available ({len(decks)} deck(s) built, none run)")
            continue
        run = run or deck_runner(runner)
        dir_ = Path(workdir) / XT_DIR
        dir_.mkdir(parents=True, exist_ok=True)
        runs: list[DeckRun] = []
        refused = None
        for s, vd, ad, d in decks:
            several = len(plan.victim.drive_ends) * len(plan.aggressors[s.aggressor].drive_ends) > 1
            stem = f"{stem_v}-{hashlib.sha256(s.aggressor.encode('utf-8')).hexdigest()[:8]}-{s.index}" + (f"-o{vd}{ad}" if several else "")
            try:
                text = build(d.ir)
                fine = analysis_command(d.ir.simulation.analyses[0], d.ir.simulation)  # type: ignore[union-attr]
                coarse_spec = d.ir.simulation.analyses[0].model_copy(update={"params": {  # type: ignore[union-attr]
                    "step": _traced(d.step * STEP_CHECK_FACTOR, "s", "tran step (self-check)"), "stop": _traced(d.stop, "s", "tran stop")}})
                coarse = analysis_command(coarse_spec, d.ir.simulation)
            except CompileError as e:
                refused = f"the deck of {_span_label(s)} does not compile: {e}"
                break
            path = dir_ / f"{stem}.cir"
            path.write_text(text, encoding="utf-8", newline="\n")
            dr = DeckRun(span=s, v_drive=vd, a_drive=ad, path=path, deck_hash=_sha(path), text=text, info=d.info)
            dr.evidence.append(Evidence(description=f"spice.xtalk deck {stem} (compiled from the IR copper and the classes' driver models)", path=str(path),
                                        content_hash=dr.deck_hash))
            run_deck(run, d, dr, [fine, coarse])
            runs.append(dr)
        if refused is not None:
            out[victim] = nv(f"{victim}: {refused}")
            continue
        status, message, measured = _judge(victim, float(budget.value), runs)
        engine_version = next((d.engine_version for d in runs if d.engine_version), None) or _version(runner)
        details = {**base, "measured": measured,
                   "decks": [{"deck": d.path.name, "span": _span_label(d.span), "aggressor": d.span.aggressor, "exact": d.span.exact,
                              "victim_driver": d.info["victim_driver"], "aggressor_driver": d.info["aggressor_driver"], "peak_rel": d.peak,
                              "problem": d.problem, "runs": d.runs, "deck_text": d.text, **d.info} for d in runs]}
        if status is S.FAIL:
            details["repair"] = "human"
        out[victim] = ValidationResult(check_id=check, status=status, message=message + f" (budget: {budget.source})", tool=runner.engine,
                                       tool_version=engine_version, artifact_hash=runs[0].deck_hash if runs else None, ir_hash=ir_hash,
                                       evidence=[e for d in runs for e in d.evidence], details=details)
    results = [out[n] for n in sorted(out, key=lambda n: order_ir.get(n, 0))]
    keep = {r.check_id for r in results}
    for check_id, last in ir.validation.latest_by_check().items():
        if check_id.startswith(CHECK_PREFIX + ".") and check_id not in keep and last.status is not S.NOT_APPLICABLE:
            results.append(ValidationResult(check_id=check_id, status=S.NOT_APPLICABLE,
                                            message="no longer a victim si.crosstalk hands to a coupled-line deck (superseded)",
                                            ir_hash=ir_hash, details={"superseded": last.status.value}))
    return results


def _version(runner: SpiceRunner) -> str:
    try:
        return runner.version()
    except ToolUnavailableError:
        return "unknown"


def xtalk_note(results: list[ValidationResult]) -> str | None:
    """The SPICE stage's one-line note about the ``spice.xtalk.*`` results (``None`` when there are none)."""
    if not results:
        return None
    count: dict[str, int] = {}
    for r in results:
        count[r.status.value] = count.get(r.status.value, 0) + 1
    return f"spice.xtalk: {len(results)} victim(s) as coupled-line decks: " + ", ".join(f"{n} {s}" for s, n in sorted(count.items()))


__all__ = [
    "CHECK_PREFIX",
    "MAX_POINTS",
    "MAX_XTALK_DECKS",
    "XT_DECK_ARC_VERSION",
    "XT_DECK_VERSION",
    "XT_DIR",
    "XT_MODEL_OMITS",
    "Budget",
    "NetLine",
    "Span",
    "VictimPlan",
    "XtDeck",
    "XtPiece",
    "deck_version",
    "nm3_budget",
    "path_delay",
    "plan_victim",
    "spice_xtalk_results",
    "victim_budget",
    "xtalk_deck",
    "xtalk_note",
]
