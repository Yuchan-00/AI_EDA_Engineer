"""Radio-design calculators: frequency plans, FM / PM modulators, coupled-resonator and crystal-ladder filters, pads, power budgets, audio filters.

Invariant: every number a radio template rests on - a stage frequency, an
image, a filter element, a fixture nominal - is the output of one of the
closed forms or exact network evaluations below, each registered like every
other calculator (the ``calc.rf.*`` / ``calc.crystal.ladder.*`` /
``calc.audio.*`` / ``calc.power.rail_budget`` / ``calc.regulator.headroom``
ids of :data:`RADIO_CALCULATORS` in :data:`~ai_eda.tools.calc.basic.ROLES` /
:data:`~ai_eda.tools.calc.basic.ROLE_UNITS` and
:data:`~ai_eda.tools.calc.recompute.CALCULATORS`, ``CALC_VERSION`` 0.10, 0.11 since the ported top-C network), so
``calc.recompute`` re-derives a stored value from the ids its provenance
names. A calculator refuses (``ValueError`` with a sentence) every input
outside its formula's domain - a tap that cannot transform, a crystal ladder
wider than its crystals allow, a negative resonator capacitor - and every
result that is not a finite number, never an extrapolated value. Beside each
``Traced`` calculator stands a plain-float helper that computes the same
number (the traced one calls it). The runtime depends on the standard library
only: every network is evaluated by complex ABCD (chain-matrix) arithmetic in
pure Python, never numpy.

Units: ``Hz``, ``s``, ``V``, ``A``, ``W``, ``ohm``, ``H``, ``F``, ``F/V``
(a varactor's slope), ``rad/V`` (a phase modulator's constant), ``deg``,
``dB``; ``None`` for an order, an index, a Q, a multiplier, a gain ratio or a
sign. A *sign* role (``side``, ``if_sign``, ``sign``) takes exactly -1 or +1
and nothing else: the categorical choice (low / high side, pre- / de-emphasis)
is written as that number so it can be traced and recomputed.

Formulas and references (each calculator's note repeats its formula):

* Frequency plan (standard superheterodyne and multiplier arithmetic, e.g.
  U. L. Rohde, J. C. Whitaker, "Communications Receivers", 3rd ed., McGraw-Hill
  2001, ch. 2). ``calc.rf.mult.stage`` f_out = M f_in (M a positive integer;
  a chain of stages is a chain of calls, so f_x prod(M_1..M_k) keeps every
  stage frequency traced); ``calc.rf.mult.spur`` f_out + k f_x (a close-in
  multiplier product, k a non-zero integer); ``calc.rf.harmonic`` k f;
  ``calc.rf.superhet.lo`` LO = f + side IF (side -1: low-side LO below the
  signal, +1: high side); ``calc.rf.superhet.image`` 2 LO - f (the other
  signal at |f - LO| = IF: LO - IF for a low-side LO, LO + IF for a high-side
  one); ``calc.rf.superhet.half_if`` (f + LO)/2 (the 2x2 response LO + IF/2 of
  a low-side LO, LO - IF/2 of a high-side one); ``calc.rf.superhet.second_image``
  f - 2 IF2 (the second mixer's image referred to the antenna, for a low-side
  first *and* second LO - neither conversion inverts the spectrum; any other
  arrangement is two ``.image`` calls at the IF); ``calc.rf.superhet.lo_spur_response``
  LO + k f_R + s IF (the signal that a spur of the LO at LO + k f_R, k a
  non-zero integer, converts to the IF; s = -1 / +1 is the sign of IF: both
  signs are responses).
* FM (J. R. Carson, 1922; the Bessel spectrum of single-tone FM, e.g.
  S. Haykin, "Communication Systems", 4th ed., 2.7): ``calc.rf.fm.obw99`` the
  99 % occupied bandwidth of a single tone, the smallest 2 n f_m with
  J_0(beta)^2 + 2 sum_{k=1..n} J_k(beta)^2 >= 0.99, beta = delta_f / f_m (a
  discrete sideband count: below beta ~ 0.14 the carrier alone holds 99 % and
  the rule gives 0 Hz). J_k is computed by Miller's backward recurrence
  normalised with J_0 + 2 sum J_2k = 1 (J. C. P. Miller 1952; Abramowitz &
  Stegun 9.12; Numerical Recipes 6.5) - stable for every beta, unlike the
  power series. Indirect FM (Armstrong): phase modulation of the integrated
  audio is frequency modulation; with a multiplier N after the modulator, an
  integrator tau_i and a modulator constant K_pm (rad/V) a clip level V_lim
  gives delta_f = N K_pm V_lim / (2 pi tau_i), so ``calc.rf.fm.pm_integrator_tau``
  tau_i = N K_pm V_lim / (2 pi delta_f) and ``calc.rf.fm.pm_drive_limit``
  V_max = 2 pi tau_i delta_f / (N K_pm).
* Varactor (the SPICE junction-capacitance law, ngspice manual, diode model
  CJO / VJ / M, reverse bias V >= 0): ``calc.rf.varactor.c_at_bias``
  C = CJO / (1 + V/VJ)^M; ``calc.rf.varactor.dc_dv`` dC/dV = -M C / (VJ + V).
* Tuned-circuit phase modulator: ``calc.rf.pm.k_pm`` the small-signal slope
  of a parallel tank's phase at resonance, K_pm = -Q_L (dC/dV) / C_tot
  (phi = -atan(Q (f/f0 - f0/f)), dphi/dC = -Q/C at f = f0; a design number
  per tank - buffered tanks on one bias node add); ``calc.rf.pm.c_fixed``
  C_fixed = C_tot - C_var(V0) - C_trim; ``calc.rf.pm.source_r`` the series
  source resistance that loads the tank to Q_L: R_s = 1/(1/(Q_L w L) -
  1/(Q_u w L)) - R_port (``.source_r_loaded`` also subtracts a load
  conductance 1/R_load; the DC block is neglected there). ``calc.rf.pm.tank_phase``
  is *not* the ideal formula (0.8-1.4 deg off the network at the KR447
  values): it is the exact phase of V(tank)/V_s of the designed network - a
  source behind R_port, a DC block C_dc, the series R_s, the tank node with
  C_fixed + C_trim + C_var to ground and L (with its series loss
  R_L = 2 pi f_q L / Q_u) returned to the bias node, whose bypass C_bypass
  and feed resistance R_feed (to the AC-grounded bias source) are in series
  with L's return; ``.tank_phase_loaded`` adds a resistive load R_load at the
  tank node (a port instead of a probe).
* Coupled resonators (M. Dishal, "Design of dissipative band-pass filters
  producing desired exact amplitude-frequency characteristics", Proc. IRE 37,
  1949; A. I. Zverev, "Handbook of Filter Synthesis", Wiley 1967): the
  Butterworth prototype g_k; coupling coefficients k_ij = 1/sqrt(g_i g_j);
  external Q_e = g_1 f0 / BW; C_res = 1/(w0^2 L). The top-C network
  (``calc.rf.resonator.top_c.*``): coupling capacitors
  C_k = k_ij (BW/f0) C_res between the resonator tops; capacitive end taps
  C_s = 1/(w0 sqrt(R_p R_t - R_t^2)) with R_p = Q_e w0 L (a tap only
  transforms down: R_t >= R_p is refused - "choose a larger L or another
  tap"), whose parallel equivalent C_s / (1 + (w0 C_s R_t)^2) is taken off the
  end resonator; shunt C_i = C_res minus the adjacent couplings (and the tap
  equivalent at an end; a non-positive C_i is refused - "choose a smaller L").
  ``.s21_db`` is the exact complex S21 of that network at f -
  20 log10 |2 V_out/V_s sqrt(R_s/R_l)| - with every inductor carrying the
  series loss R = w0 L / Q_u fixed at f0 (the fixture runner's ``loss_q``
  model); ``.rel_s21_db`` is s21(f) - s21(f_ref). These, not the symmetric
  narrowband formulas below, are the fixture nominals of every tank and top-C
  filter: at -/+ 8-33 % offsets the + side of a top-C network is 5-12 dB
  weaker than the formula. ``.bw_for_qe`` BW = g_1 f0 / Q_e (a tank specified
  by its end Q). ``calc.rf.q_parallel`` Q = R / (w L).
  A network between *loaded* ports (``CALC_VERSION`` 0.11; a multiplier
  tank whose source port carries the collector feed choke to AC ground and
  whose load port the next stage's base divider): ``.port_r`` / ``.port_x``
  the series equivalent R' + j X' of Z = R_port // (j w0 L_p + w0 L_p / Q_p)
  at f0 (the choke with the fixture runner's loss model, fixed at f0);
  ``.c_tap_reactive`` the source tap that makes the end resonator see
  R_p = Q_e w0 L through R' + j X': C_s = 1/(w0 (X' + sqrt(R' (R_p - R')))) -
  the parallel equivalent it takes off the end resonator is the resistive
  tap's for R' (so ``.c_shunt`` with r_source = R' and r_load = the load the
  network sees, R_load // the divider, is the network's shunt capacitor);
  ``.ported_s21_db`` / ``.ported_rel_s21_db`` the exact S21 of that network
  with the shunt L_p (loss w0 L_p / Q_p) at the source port and the extra
  shunt conductance 1 / r_load_eff - 1 / R_load at the load port, normalised
  to the two *port* resistances (the fixture's own S21 definition). A tap
  designed for R_port alone mistunes the end resonator: a 1 uH choke on a
  1 kohm port at 223.8 MHz cost the kr447 x6 tank 2-3 dB of rejection per
  side on ngspice-42.
* Crystal ladder (lower-sideband ladder: crystals in series, shunt coupling
  capacitors, series mesh-tuning capacitors; Dishal; W. Hayward, R. Campbell,
  B. Larkin, "Experimental Methods in RF Design" (EMRFD), ARRL 2003, ch. 3):
  ``.k`` k_{i,i+1} = 1/sqrt(g_i g_{i+1}) and ``.q`` q = g_1 (Butterworth).
  The classical EMRFD equations C_ij = C_m f_0 / (k_ij BW) and
  R_end = 2 pi BW L_m / q ignore the holder capacitance C0; with the KR447
  crystal model (C_m 6 fF, C0 4 pF) a 7.5 kHz design from them measured
  3.4 kHz on ngspice-42. The C0-aware design used here is the same Dishal
  method on reactance slope parameters (G. L. Matthaei, L. Young,
  E. M. T. Jones, "Microwave Filters, Impedance-Matching Networks, and
  Coupling Structures", 1964, 8.02 - series resonators coupled by
  K-inverters): a shunt capacitor C_ij is a K-inverter K = 1/(w0 C_ij) whose
  negative series arms are the series capacitors of the two adjacent meshes;
  every mesh (crystal + its series capacitors) resonates at the common f0 and
  has the reactance slope parameter x = (w0/2) dX_crystal/dw + X_c/2, where
  the crystal's reactance with C0 across the motional arm is
  X_c = X_m / (1 - w C0 X_m), X_m = w L_m - 1/(w C_m), and
  dX_c/dw = (L_m + 1/(w^2 C_m) + C0 X_m^2) / (1 - w C0 X_m)^2. Then
  K_ij = w_f x k_ij (w_f = BW/f0), C_ij = 1/(w0 K_ij), R_end = w_f x / g_1,
  f0 solves X_c(f0) = w_f x kappa (kappa = the largest sum of a mesh's
  adjacent k), the mesh(es) with that sum carry no tuning capacitor and every
  other mesh i the series capacitor 1/(w0 (X_c - K_{i-1,i} - K_{i,i+1}))
  (``.mesh_c``). The lower of the two roots is the design (it is the classical
  one as C0 -> 0; the equations reduce to the EMRFD ones for C0 = 0 and
  f0 -> f_s). The equation has no root when the ladder is wider than its
  crystals allow - about BW_max = f_s C_m / (4 C0 kappa) - and the calculator
  refuses with that bound instead of returning a design that cannot work
  (measured on ngspice-42: every 6-pole design from the classical formulas
  with C_m 6 fF / C0 4 pF stayed below 5 kHz whatever bandwidth it targeted).
  ``.s21_db`` is the exact response of the designed ladder between R_end at
  both ends with every crystal's motional R_m included; ``.center`` is the
  midpoint of its half-power (10 log10 2 dB below the peak) edges - the
  passband centre a fixture is judged around (C0 skews the response, so the
  mesh frequency f0 is not it). The C0-aware design meets its bandwidth
  only for *lossless* crystals: with R_m -> 0 (``.s21_db``, the calculator's
  exact response; ngspice cannot run R_m = 0, its operating point is
  singular) the -3 dB bandwidth (10 log10 2 below the peak) came within
  +0.75 .. +2.2 % of the target for the 6-pole 2.4 / 3.0 / 3.5 kHz ladders
  (C_m 6 fF) and the 7.5 kHz one (C_m 18 fF), +2.2 % for a 4-pole 7.5 kHz
  one (C_m 12 fF), all with C0 4 pF (+5.9 % at 4.4 kHz, next to the C0
  bound; exact for C0 = 0). With the motional loss of the design's model
  (``model.xtal21.rm`` 25 ohm) the realised bandwidth is about 4-14 %
  *narrower* than the target (6-pole 6 fF: 2055 Hz for 2.4 kHz, -14.4 %;
  3177 Hz for 3.5 kHz, -9.2 %; 4213 Hz for 4.4 kHz, -4.3 %; 6-pole 18 fF:
  6465 Hz for 7.5 kHz, -13.8 %; 4-pole 12 fF: -3.8 %), with 2.5-5.9 dB of
  passband loss; ngspice-42 reproduces ``.s21_db`` with R_m within 0.01 dB.
  The classical formulas gave 3.85 kHz for 7.5 kHz lossless and 3.44 kHz with
  R_m 25 ohm (-49 / -54 %). A template therefore judges a ladder's realised
  bandwidth by ``.s21_db`` / ``.center`` with R_m included (or pre-distorts
  its target), never by the design bandwidth it asked for.
* Pads and matches: ``calc.rf.attenuator.pi.r_shunt`` Z (K + 1)/(K - 1) and
  ``.r_series`` Z (K - 1/K)/2, K = 10^(A/20) (the matched symmetric pi pad,
  e.g. "Reference Data for Radio Engineers", ITT, ch. 10);
  ``calc.rf.pa.load_line_r`` (V_CC - V_sat)^2 / (2 P) (S. C. Cripps, "RF Power
  Amplifiers for Wireless Communications", 2nd ed., Artech 2006, ch. 2);
  ``calc.rf.quarter_wave_lumped.l`` Z0/w and ``.c`` 1/(w Z0) (the C-L-C pi
  equivalent of a lossless lambda/4 line of Z0 at one frequency).
* Quadrature detector tank (circuit analysis): ``calc.rf.quad.phase`` the
  phase of V(quad)/V_s for a source R_s driving the coupling capacitor C_q
  into the parallel tank L (series loss 2 pi f_q L / Q_u) // (C_p + C_trim) //
  R_p - exact, so the tank's detuning by C_q itself is included.
* Bias and levels (standard): ``calc.rf.bjt_bias.ic`` I_C ~ I_E =
  (V_B - V_BE)/R_E (beta -> infinity; V_B from an unloaded divider,
  ``calc.divider.v_out``; V_BE a stated choice); ``calc.rf.limiter.level``
  V_D x gain (the diode drop a stated choice); ``calc.rf.db_sum`` a + b (two
  levels or gains in dB, so a stage response composed of first-order terms
  stays a calculator output).
* Audio (standard first-order networks): ``calc.audio.emphasis.corner``
  f_c = 1/(2 pi tau); ``calc.audio.emphasis.db_at`` sign x 10 log10(1 +
  (f/f_c)^2) (+1 pre-emphasis, -1 de-emphasis); ``calc.audio.highpass1.db_at``
  -10 log10(1 + (f_c/f)^2); ``calc.audio.lowpass1.db_at`` -10 log10(1 +
  (f/f_c)^2); ``calc.audio.lossy_integrator.db_at`` the inverting integrator
  of time constant tau_i = R_in C with a DC-limit resistor R_dc across C:
  |H| = (R_dc/R_in) / sqrt(1 + (w tau_i R_dc/R_in)^2) (-> 1/(w tau_i) as
  R_dc -> infinity); ``calc.audio.butterworth.q`` the k-th pole pair's
  Q_k = 1/g_k = 1/(2 sin((2k - 1) pi/(2n))), k = 1 .. floor(n/2) (1.3066 and
  0.5412 for n = 4; A. B. Williams, F. J. Taylor, "Electronic Filter Design
  Handbook"); ``calc.audio.sallen_key.c1`` / ``.c2`` the unity-gain Sallen-Key
  low-pass with R1 = R2 = R: C1 = 2 Q/(w_c R) (feedback), C2 = 1/(2 Q w_c R)
  (to ground) (R. P. Sallen, E. L. Key, IRE Trans. CT-2, 1955).
* Time-out timer: ``calc.rf.tot.period`` 2^13 k_RC R C - the 4060's Q14 first
  rises after 2^13 oscillator periods; the RC-oscillator period k_RC R C is a
  datasheet fact, so k_RC is a stated choice [UNVERIFIED until grounded].
* Filter loss and tuned-circuit ceilings: ``calc.rf.bpf.dissipation_loss``
  L0 = (10/ln 10) sum(g_i) f0 / (BW Q_u) dB (S. B. Cohn, "Dissipation loss in
  multiple-coupled-resonator filters", Proc. IRE 47, 1959; Butterworth g;
  small-loss approximation - checked against ngspice-42 Q-40 top-C decks:
  3.44 vs 3.37 dB, 9.72 vs 9.30 dB, 4.34 vs 4.21 dB);
  ``calc.rf.resonator.single_tuned.rejection`` 10 log10(1 + x^2) and
  ``.double_tuned.rejection`` 10 log10(1 + x^4/4) (a critically coupled pair
  of equal loaded Q), x = Q_L (f/f0 - f0/f) - symmetric narrowband ceilings,
  never a fixture nominal; ``.single_tuned.insertion_loss``
  -20 log10(1 - Q_L/Q_u) for one resonator whose total loaded Q_L includes
  Q_u (a critically coupled pair's loss is Cohn's, not this applied twice).
* Power (definitions): ``calc.power.rail_budget`` the rating margin
  I_rating - I_load (negative: the load exceeds the rating - a number, not a
  refusal); ``calc.regulator.headroom`` V_in,min - I_load R_path - V_dropout -
  V_out (negative: out of regulation at the minimum input). Every current and
  dropout is a datasheet fact or a stated choice.

Not modelled anywhere here: device nonlinearity (a multiplier's harmonic
generation, the PA, the limiter's waveform), the parasitics of real parts
(self-resonance, pad and track inductance, ground returns), a crystal's
spurious modes and a varactor's series resistance, noise, and temperature.
A number from this module is a design target or the response of the
schematic network under the stated model values; only a measurement says
what a built board does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ai_eda.ir.provenance import Traced
from ai_eda.tools.calc.basic import _derived
from ai_eda.tools.calc.rf import (
    _finite,
    _integer,
    _log10_positive,
    _non_negative,
    _nonzero_result,
    _num,
    _positive,
    _pow10,
    butterworth_g_values,
)

#: dB per neper of power, 10 / ln 10 = 4.3429... (Cohn's "4.343")
DB_PER_NEPER = 10.0 / math.log(10.0)
#: the half-power level below the peak that defines a -3 dB edge: 10 log10 2 = 3.0103 dB
HALF_POWER_DB = 10.0 * math.log10(2.0)
#: highest order of a coupled-resonator (top-C) network the calculators design
RESONATOR_MAX_ORDER = 10
#: orders of a crystal ladder the calculators design (a ladder needs at least two coupled crystals)
LADDER_MIN_ORDER = 2
LADDER_MAX_ORDER = 12
#: widest fractional bandwidth BW/f0 the narrowband coupled-resonator design is used for
MAX_FRACTIONAL_BW = 0.5
#: highest FM index calc.rf.fm.obw99 evaluates (a narrowband radio's beta is a few units)
OBW_MAX_INDEX = 1000.0
#: the share of the power the occupied bandwidth holds
OBW_POWER_FRACTION = 0.99
#: oscillator periods until the 4060's Q14 first rises (2^13)
TOT_Q14_PERIODS = 2 ** 13
#: largest multiplier / harmonic / spur order the frequency-plan calculators accept
MAX_PLAN_ORDER = 1000
#: a mesh whose tuning reactance is below this share of the crystal reactance needs no tuning capacitor
_MESH_TUNE_TOL = 1e-9


# --------------------------------------------------------------------------- small checks


def _sign(x: float, what: str) -> int:
    """A sign role: exactly -1 or +1."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or float(x) not in (-1.0, 1.0):
        raise ValueError(f"{what} must be -1 or +1, got {x!r}")
    return int(x)


def _nonzero_integer(x: float, what: str, lo: int, hi: int) -> int:
    n = _integer(x, what, lo, hi)
    if n == 0:
        raise ValueError(f"{what} must not be 0")
    return n


def _positive_frequency_result(f: float, tool: str, what: str) -> float:
    if not math.isfinite(f):
        raise ValueError(f"{tool} overflows: {what} is not a finite number for these inputs")
    if f <= 0.0:
        raise ValueError(f"{tool}: {what} = {f:.9g} Hz is not a positive frequency for these inputs")
    return f


def _omega(f_hz: float, what: str = "the frequency") -> float:
    return 2.0 * math.pi * _positive(f_hz, what)


def _db_mag(h: complex, tool: str) -> float:
    m = abs(h)
    if not math.isfinite(m):
        raise ValueError(f"{tool} overflows: the network's transfer is not a finite number at this frequency")
    if m == 0.0:
        raise ValueError(f"{tool} underflows: the network's transfer rounds to 0 at this frequency")
    return 20.0 * math.log10(m)


# --------------------------------------------------------------------------- chain-matrix (ABCD) network evaluation


def _cascade(elements: list[tuple[str, complex]]) -> tuple[complex, complex, complex, complex]:
    """The ABCD matrix of a ladder of ``("series", Z)`` impedances and ``("shunt", Y)`` admittances, source side first."""
    a, b, c, d = 1 + 0j, 0j, 0j, 1 + 0j
    for kind, v in elements:
        if kind == "series":
            a, b, c, d = a, a * v + b, c, c * v + d
        else:
            a, b, c, d = a + b * v, b, c + d * v, d
    return a, b, c, d


def _s21(elements: list[tuple[str, complex]], r_source: float, r_load: float, tool: str) -> complex:
    """S21 = 2 V_out/V_s sqrt(R_s/R_l) of the ladder between a source R_s and a load R_l."""
    a, b, c, d = _cascade(elements)
    den = a + b / r_load + c * r_source + d * r_source / r_load
    if den == 0 or not (math.isfinite(den.real) and math.isfinite(den.imag)):
        raise ValueError(f"{tool} overflows: the network's chain matrix is not finite at this frequency")
    return 2.0 * math.sqrt(r_source / r_load) / den


def _open_transfer(elements: list[tuple[str, complex]], tool: str) -> complex:
    """V_out / V_s of a ladder whose source resistance is its first series element and whose output is open (a probe)."""
    a = _cascade(elements)[0]
    if a == 0 or not (math.isfinite(a.real) and math.isfinite(a.imag)):
        raise ValueError(f"{tool} overflows: the network's chain matrix is not finite at this frequency")
    return 1.0 / a


def _cap_z(w: float, c: float) -> complex:
    return 1.0 / (1j * w * c)


# --------------------------------------------------------------------------- frequency plan (plain helpers)


def multiplier_stage_hz(f_in: float, m: float) -> float:
    """A multiplier stage's output f_out = M f_in (M a positive integer)."""
    mult = _integer(m, "the multiplication factor M", 1, MAX_PLAN_ORDER)
    return _nonzero_result(_positive(f_in, "the input frequency") * mult, "calc.rf.mult.stage")


def multiplier_spur_hz(f_out: float, f_x: float, k: float) -> float:
    """A close-in multiplier product f_out + k f_x (k a non-zero integer)."""
    order = _nonzero_integer(k, "the spur order k", -MAX_PLAN_ORDER, MAX_PLAN_ORDER)
    f = _positive(f_out, "the output frequency") + order * _positive(f_x, "the reference frequency")
    return _positive_frequency_result(f, "calc.rf.mult.spur", "f_out + k f_x")


def harmonic_hz(f: float, k: float) -> float:
    """The k-th harmonic k f (k a positive integer)."""
    order = _integer(k, "the harmonic number k", 1, MAX_PLAN_ORDER)
    return _nonzero_result(_positive(f, "the frequency") * order, "calc.rf.harmonic")


def superhet_lo_hz(f: float, f_if: float, side: float) -> float:
    """LO = f + side IF (side -1: low-side LO, +1: high-side LO)."""
    s = _sign(side, "the LO side (-1 low, +1 high)")
    lo = _positive(f, "the signal frequency") + s * _positive(f_if, "the IF")
    return _positive_frequency_result(lo, "calc.rf.superhet.lo", "the LO")


def superhet_image_hz(f: float, lo: float) -> float:
    """The image 2 LO - f: the other signal that mixes with LO to the same IF |f - LO|."""
    fs, flo = _positive(f, "the signal frequency"), _positive(lo, "the LO frequency")
    if fs == flo:
        raise ValueError("an LO at the signal frequency converts to 0 Hz: there is no IF and no image")
    return _positive_frequency_result(2.0 * flo - fs, "calc.rf.superhet.image", "the image 2 LO - f")


def superhet_half_if_hz(f: float, lo: float) -> float:
    """The half-IF (2x2) response (f + LO)/2 = LO +/- IF/2 on the signal's side of the LO."""
    fs, flo = _positive(f, "the signal frequency"), _positive(lo, "the LO frequency")
    if fs == flo:
        raise ValueError("an LO at the signal frequency converts to 0 Hz: there is no IF and no half-IF response")
    return (fs + flo) / 2.0


def superhet_second_image_hz(f: float, f_if2: float) -> float:
    """The second IF's image referred to the antenna, f - 2 IF2 (low-side first and second LO)."""
    return _positive_frequency_result(_positive(f, "the signal frequency") - 2.0 * _positive(f_if2, "the second IF"), "calc.rf.superhet.second_image", "f - 2 IF2")


def superhet_lo_spur_response_hz(lo: float, f_r: float, k: float, f_if: float, if_sign: float) -> float:
    """The signal an LO spur at LO + k f_R converts to the IF: LO + k f_R + s IF (k a non-zero integer, s = -1 or +1)."""
    order = _nonzero_integer(k, "the LO spur order k", -MAX_PLAN_ORDER, MAX_PLAN_ORDER)
    s = _sign(if_sign, "the sign of the IF (-1 or +1)")
    f = _positive(lo, "the LO frequency") + order * _positive(f_r, "the LO reference frequency") + s * _positive(f_if, "the IF")
    return _positive_frequency_result(f, "calc.rf.superhet.lo_spur_response", "LO + k f_R + s IF")


# --------------------------------------------------------------------------- FM, PM, varactor (plain helpers)


def bessel_j_values(beta: float, n_max: int) -> list[float]:
    """[J_0(beta) .. J_n_max(beta)] by Miller's backward recurrence, normalised with J_0 + 2 sum J_2k = 1."""
    x = _non_negative(beta, "the modulation index beta")
    n_top = _integer(n_max, "the highest Bessel order", 0, 100_000)
    if x == 0.0:
        return [1.0] + [0.0] * n_top
    start = max(n_top, int(x)) + int(math.sqrt(40.0 * (max(n_top, x) + 1.0))) + 20
    start += start % 2  # even, so the normalisation sum J_0 + 2 sum J_2k covers every order computed
    a = [0.0] * (start + 2)  # a[start + 1] = 0, a[start] = 1: unnormalised J_k by backward recurrence
    a[start] = 1.0
    for k in range(start, 0, -1):
        a[k - 1] = 2.0 * k / x * a[k] - a[k + 1]
        if abs(a[k - 1]) > 1e100:  # keep the unnormalised values finite; the normalisation removes the scale
            for i in range(k - 1, start + 1):
                a[i] *= 1e-100
    norm = a[0] + 2.0 * sum(a[k] for k in range(2, start + 1, 2))
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError(f"the Bessel recurrence did not normalise for beta = {x:.6g}")
    return [v / norm for v in a[: n_top + 1]]


def fm_obw99_hz(delta_f: float, f_m: float) -> float:
    """The 99 % occupied bandwidth of single-tone FM: the smallest 2 n f_m holding 99 % of the power (Bessel sidebands)."""
    df = _non_negative(delta_f, "the peak deviation")
    fm = _positive(f_m, "the modulating frequency")
    beta = df / fm
    if not math.isfinite(beta) or beta > OBW_MAX_INDEX:
        raise ValueError(f"the modulation index beta = {beta:.6g} is above the {OBW_MAX_INDEX:g} this sideband count evaluates")
    n_max = int(beta + 10.0 * math.sqrt(beta + 1.0) + 20.0)
    j = bessel_j_values(beta, n_max)
    total = j[0] * j[0]
    n = 0
    while total < OBW_POWER_FRACTION:
        n += 1
        if n > n_max:
            raise ValueError(f"the Bessel sideband sum did not reach {OBW_POWER_FRACTION:g} within {n_max} sidebands for beta = {beta:.6g}")
        total += 2.0 * j[n] * j[n]
    return 2.0 * n * fm


def _n_mult(n: float) -> int:
    return _integer(n, "the multiplication factor N", 1, MAX_PLAN_ORDER)


def pm_integrator_tau_s(n_mult: float, k_pm: float, v_lim: float, delta_f: float) -> float:
    """tau_i = N K_pm V_lim / (2 pi delta_f): the integrator that turns a clipped level V_lim into the peak deviation."""
    n = _n_mult(n_mult)
    tau = n * _positive(k_pm, "the modulator constant K_pm") * _positive(v_lim, "the clip level") / (2.0 * math.pi * _positive(delta_f, "the peak deviation"))
    return _nonzero_result(tau, "calc.rf.fm.pm_integrator_tau")


def pm_drive_limit_v(n_mult: float, k_pm: float, tau_i: float, delta_f: float) -> float:
    """V_max = 2 pi tau_i delta_f / (N K_pm): the largest integrator input that stays within the peak deviation."""
    n = _n_mult(n_mult)
    v = 2.0 * math.pi * _positive(tau_i, "the integrator time constant") * _positive(delta_f, "the peak deviation") / (n * _positive(k_pm, "the modulator constant K_pm"))
    return _nonzero_result(v, "calc.rf.fm.pm_drive_limit")


def _varactor_inputs(cjo: float, vj: float, m: float, v: float) -> tuple[float, float, float, float]:
    c0 = _positive(cjo, "the zero-bias capacitance CJO")
    phi = _positive(vj, "the junction potential VJ")
    grading = _positive(m, "the grading coefficient M")
    bias = _finite(v, "the reverse bias")
    if bias < 0.0:
        raise ValueError(f"the reverse bias must not be negative (a forward-biased varactor is outside this law's use), got {v!r}")
    return c0, phi, grading, bias


def varactor_c_f(cjo: float, vj: float, m: float, v: float) -> float:
    """C = CJO / (1 + V/VJ)^M at the reverse bias V >= 0 (SPICE junction law)."""
    c0, phi, grading, bias = _varactor_inputs(cjo, vj, m, v)
    # (1 + V/VJ)^M as exp(M log1p(V/VJ)): M > 0 and V >= 0, so the exponent is <= 0 and exp never overflows (a float ** would
    # raise OverflowError for M = 5000); a denominator past the float range gives C = 0.0, refused below as an underflow
    return _nonzero_result(c0 * math.exp(-grading * math.log1p(bias / phi)), "calc.rf.varactor.c_at_bias")


def varactor_slope_f_per_v(cjo: float, vj: float, m: float, v: float) -> float:
    """dC/dV = -M C / (VJ + V) at the reverse bias V >= 0 (F/V, negative)."""
    c0, phi, grading, bias = _varactor_inputs(cjo, vj, m, v)
    c = varactor_c_f(c0, phi, grading, bias)
    return _nonzero_result(-grading * c / (phi + bias), "calc.rf.varactor.dc_dv")


def pm_k_pm_rad_per_v(q_l: float, dc_dv: float, c_tot: float) -> float:
    """K_pm = -Q_L (dC/dV) / C_tot (rad/V): a parallel tank's small-signal phase slope at resonance."""
    slope = _finite(dc_dv, "the varactor slope dC/dV")
    if slope == 0.0:
        raise ValueError("a varactor slope dC/dV of 0 modulates nothing: K_pm would be 0")
    return -_positive(q_l, "the loaded Q") * slope / _positive(c_tot, "the tank capacitance C_tot")


def pm_c_fixed_f(c_tot: float, c_var: float, c_trim: float) -> float:
    """C_fixed = C_tot - C_var(V0) - C_trim: the fixed capacitor that completes the tank."""
    c = _positive(c_tot, "the tank capacitance C_tot") - _positive(c_var, "the varactor capacitance") - _non_negative(c_trim, "the trimmer capacitance")
    if c <= 0.0:
        raise ValueError(f"the varactor and trimmer already exceed C_tot = {c_tot:.6g} F: no fixed capacitor completes the tank (choose a larger C_tot)")
    return c


def _pm_source(f: float, l: float, q_l: float, q_u: float, r_port: float, g_load: float) -> float:
    xl = _omega(f) * _positive(l, "the tank inductance")
    ql, qu = _positive(q_l, "the loaded Q"), _positive(q_u, "the inductor's Q_u")
    if ql >= qu:
        raise ValueError(f"the loaded Q {ql:.6g} must stay below the inductor's unloaded Q {qu:.6g}")
    g_src = 1.0 / (ql * xl) - 1.0 / (qu * xl) - g_load
    if g_src <= 0.0:
        raise ValueError(f"the tank's own loss and load already bring it below Q_L = {ql:.6g}: no source resistance is left")
    r_s = 1.0 / g_src - _non_negative(r_port, "the port resistance")
    if r_s <= 0.0:
        raise ValueError(f"the port resistance {r_port:.6g} ohm alone loads the tank below Q_L = {ql:.6g}: no series source resistance is left")
    return r_s


def pm_source_r_ohm(f: float, l: float, q_l: float, q_u: float, r_port: float) -> float:
    """R_s = 1/(1/(Q_L w L) - 1/(Q_u w L)) - R_port: the series source resistance that loads the tank to Q_L."""
    return _pm_source(f, l, q_l, q_u, r_port, 0.0)


def pm_source_r_loaded_ohm(f: float, l: float, q_l: float, q_u: float, r_port: float, r_load: float) -> float:
    """R_s = 1/(1/(Q_L w L) - 1/(Q_u w L) - 1/R_load) - R_port (the tank node also loaded by R_load)."""
    return _pm_source(f, l, q_l, q_u, r_port, 1.0 / _positive(r_load, "the load resistance"))


def _pm_transfer(f: float, f_q: float, l: float, q_u: float, c_fixed: float, c_trim: float, c_var: float, r_port: float,
                 c_dc: float, r_s: float, c_bypass: float, r_feed: float, g_load: float, tool: str) -> complex:
    w = _omega(f)
    ind = _positive(l, "the tank inductance")
    r_l = _omega(f_q, "the frequency Q_u is stated at") * ind / _positive(q_u, "the inductor's Q_u")
    c_tank = _non_negative(c_fixed, "the fixed capacitance") + _non_negative(c_trim, "the trimmer capacitance") + _positive(c_var, "the varactor capacitance")
    z_ret = 1.0 / (1j * w * _positive(c_bypass, "the bias bypass capacitance") + 1.0 / _positive(r_feed, "the bias feed resistance"))
    y_tank = 1j * w * c_tank + 1.0 / (1j * w * ind + r_l + z_ret) + g_load
    z_ser = _non_negative(r_port, "the port resistance") + _cap_z(w, _positive(c_dc, "the DC block")) + _non_negative(r_s, "the series source resistance")
    return _open_transfer([("series", z_ser), ("shunt", y_tank)], tool)


def pm_tank_phase_deg(f: float, f_q: float, l: float, q_u: float, c_fixed: float, c_trim: float, c_var: float, r_port: float,
                      c_dc: float, r_s: float, c_bypass: float, r_feed: float) -> float:
    """The exact phase of V(tank)/V_s of the designed phase-modulator tank (the tank node probed, not loaded), degrees."""
    h = _pm_transfer(f, f_q, l, q_u, c_fixed, c_trim, c_var, r_port, c_dc, r_s, c_bypass, r_feed, 0.0, "calc.rf.pm.tank_phase")
    return math.degrees(math.atan2(h.imag, h.real))


def pm_tank_phase_loaded_deg(f: float, f_q: float, l: float, q_u: float, c_fixed: float, c_trim: float, c_var: float, r_port: float,
                             c_dc: float, r_s: float, c_bypass: float, r_feed: float, r_load: float) -> float:
    """As :func:`pm_tank_phase_deg` with a resistive load R_load at the tank node (a port), degrees."""
    g = 1.0 / _positive(r_load, "the load resistance")
    h = _pm_transfer(f, f_q, l, q_u, c_fixed, c_trim, c_var, r_port, c_dc, r_s, c_bypass, r_feed, g, "calc.rf.pm.tank_phase_loaded")
    return math.degrees(math.atan2(h.imag, h.real))


# --------------------------------------------------------------------------- coupled-resonator (top-C) networks (plain helpers)


def q_parallel_value(f: float, l: float, r: float) -> float:
    """The Q of a parallel R with L at f: R / (w L)."""
    return _nonzero_result(_positive(r, "the parallel resistance") / (_omega(f) * _positive(l, "the inductance")), "calc.rf.q_parallel")


def _narrowband(f0: float, bw: float) -> tuple[float, float]:
    fc, b = _positive(f0, "the centre frequency"), _positive(bw, "the bandwidth")
    if b / fc > MAX_FRACTIONAL_BW:
        raise ValueError(f"a bandwidth of {b:.6g} Hz at {fc:.6g} Hz is {100 * b / fc:.3g} % of the centre: the coupled-resonator design is a narrowband method (at most {100 * MAX_FRACTIONAL_BW:g} %)")
    return fc, b


def _prototype(n: float, lo: int, hi: int, what: str) -> tuple[int, list[float]]:
    order = _integer(n, what, lo, hi)
    return order, butterworth_g_values(order)[:order]


def top_c_bw_for_qe_hz(n: float, f0: float, q_e: float) -> float:
    """BW = g_1 f0 / Q_e: the bandwidth of the Butterworth top-C network whose end resonators have the external Q Q_e."""
    order, g = _prototype(n, 1, RESONATOR_MAX_ORDER, "the resonator count n")
    return _nonzero_result(g[0] * _positive(f0, "the centre frequency") / _positive(q_e, "the external Q"), "calc.rf.resonator.top_c.bw_for_qe")


@dataclass(frozen=True)
class TopCNetwork:
    """A designed Butterworth top-C coupled-resonator network (Dishal / Zverev).

    ``c_couple`` holds the n - 1 top coupling capacitors, ``c_shunt`` the n
    resonator shunt capacitors (after the couplings and the taps' parallel
    equivalents are taken off C_res), ``c_tap_source`` / ``c_tap_load`` the
    series end taps, ``r_p_source`` / ``r_p_load`` the end resonators'
    required parallel resistance Q_e w0 L. Farads, ohms, henries.
    """

    n: int
    f0: float
    bw: float
    l: float
    r_source: float
    r_load: float
    c_res: float
    q_e: float
    r_p_source: float
    r_p_load: float
    c_tap_source: float
    c_tap_load: float
    c_couple: tuple[float, ...]
    c_shunt: tuple[float, ...]


def _tap(q_e: float, w0: float, l: float, r_t: float, end: str) -> tuple[float, float, float]:
    r_p = q_e * w0 * l
    if r_t >= r_p:
        raise ValueError(f"the {end} termination {r_t:.6g} ohm is not below the end resonator's R_p = Q_e w0 L = {r_p:.6g} ohm: a capacitive tap only transforms down - choose a larger L or another tap")
    c_s = 1.0 / (w0 * math.sqrt(r_p * r_t - r_t * r_t))
    x = w0 * c_s * r_t  # sqrt(R_t / (R_p - R_t)): bounded, but a product like every other power in this module
    c_eq = c_s / (1.0 + x * x)
    return r_p, c_s, c_eq


def top_c_network(n: float, f0: float, bw: float, l: float, r_source: float, r_load: float) -> TopCNetwork:
    """Design the Butterworth top-C network of ``n`` resonators of inductance ``l`` at f0 / BW between R_source and R_load."""
    order, g = _prototype(n, 1, RESONATOR_MAX_ORDER, "the resonator count n")
    fc, b = _narrowband(f0, bw)
    ind = _positive(l, "the resonator inductance")
    rs, rl = _positive(r_source, "the source termination"), _positive(r_load, "the load termination")
    w0 = 2.0 * math.pi * fc
    c_res = 1.0 / (w0 * w0 * ind)
    frac = b / fc
    k = [1.0 / math.sqrt(g[i] * g[i + 1]) for i in range(order - 1)]
    cc = [ki * frac * c_res for ki in k]
    q_e = g[0] * fc / b  # Butterworth: g_n = g_1, both ends alike
    rp_s, cs_s, ceq_s = _tap(q_e, w0, ind, rs, "source")
    rp_l, cs_l, ceq_l = _tap(q_e, w0, ind, rl, "load")
    shunt = []
    for i in range(order):
        c = c_res - (cc[i - 1] if i > 0 else 0.0) - (cc[i] if i < order - 1 else 0.0)
        if i == 0:
            c -= ceq_s
        if i == order - 1:
            c -= ceq_l
        if not c > 0.0:
            raise ValueError(f"resonator {i + 1}'s shunt capacitor would be {c:.6g} F: the couplings and taps exceed C_res = 1/(w0^2 L) = {c_res:.6g} F - choose a smaller L")
        shunt.append(c)
    return TopCNetwork(n=order, f0=fc, bw=b, l=ind, r_source=rs, r_load=rl, c_res=c_res, q_e=q_e, r_p_source=rp_s, r_p_load=rp_l,
                       c_tap_source=cs_s, c_tap_load=cs_l, c_couple=tuple(cc), c_shunt=tuple(shunt))


def top_c_s21_complex(net: TopCNetwork, q_u: float, f: float) -> complex:
    """The exact S21 of a designed top-C network at f, every inductor with the series loss w0 L / Q_u."""
    w = _omega(f)
    r_q = 2.0 * math.pi * net.f0 * net.l / _positive(q_u, "the inductor's Q_u")
    elements: list[tuple[str, complex]] = [("series", _cap_z(w, net.c_tap_source))]
    for i in range(net.n):
        elements.append(("shunt", 1j * w * net.c_shunt[i] + 1.0 / (r_q + 1j * w * net.l)))
        if i < net.n - 1:
            elements.append(("series", _cap_z(w, net.c_couple[i])))
    elements.append(("series", _cap_z(w, net.c_tap_load)))
    return _s21(elements, net.r_source, net.r_load, "calc.rf.resonator.top_c.s21_db")


def top_c_s21_db_value(n: float, f0: float, bw: float, l: float, r_source: float, r_load: float, q_u: float, f: float) -> float:
    """20 log10 |S21| (dB) of the designed top-C network at f."""
    net = top_c_network(n, f0, bw, l, r_source, r_load)
    return _db_mag(top_c_s21_complex(net, q_u, f), "calc.rf.resonator.top_c.s21_db")


def top_c_rel_s21_db_value(n: float, f0: float, bw: float, l: float, r_source: float, r_load: float, q_u: float, f: float, f_ref: float) -> float:
    """s21_db(f) - s21_db(f_ref) of the designed top-C network (a rejection relative to the passband)."""
    net = top_c_network(n, f0, bw, l, r_source, r_load)
    tool = "calc.rf.resonator.top_c.rel_s21_db"
    return _db_mag(top_c_s21_complex(net, q_u, f), tool) - _db_mag(top_c_s21_complex(net, q_u, f_ref), tool)


# --------------------------------------------------------------------------- a top-C network between loaded ports (a collector choke, a base divider)


def port_shunt_l_impedance(r_port: float, l_port: float, q_port: float, f0: float) -> complex:
    """R_port // (w0 L_p / Q_p + j w0 L_p) at f0: a port resistance shunted by an inductor (a collector feed choke to AC ground) whose series
    loss is the fixture runner's ``loss_q`` model, fixed at f0."""
    rp = _positive(r_port, "the port resistance")
    lp = _positive(l_port, "the port's shunt inductance")
    q = _positive(q_port, "the shunt inductor's Q")
    w0 = _omega(f0, "the centre frequency")
    zb = complex(w0 * lp / q, w0 * lp)
    return rp * zb / (rp + zb)


def top_c_port_r_ohm(r_port: float, l_port: float, q_port: float, f0: float) -> float:
    """Re Z of R_port // (L_p with its loss) at f0 (ohm): the resistance the network's source tap transforms."""
    return port_shunt_l_impedance(r_port, l_port, q_port, f0).real


def top_c_port_x_ohm(r_port: float, l_port: float, q_port: float, f0: float) -> float:
    """Im Z of R_port // (L_p with its loss) at f0 (ohm, inductive > 0): the reactance the source tap absorbs."""
    return port_shunt_l_impedance(r_port, l_port, q_port, f0).imag


def _tap_reactive(q_e: float, w0: float, l: float, r_t: float, x_t: float, end: str) -> float:
    """The series tap into a termination R_t + j X_t: 1 / (w0 (X_t + sqrt(R_t (R_p - R_t)))), so the end resonator sees R_p = Q_e w0 L."""
    r_p = q_e * w0 * l
    if r_t >= r_p:
        raise ValueError(f"the {end} termination's resistance {r_t:.6g} ohm is not below the end resonator's R_p = Q_e w0 L = {r_p:.6g} ohm: a capacitive "
                         "tap only transforms down - choose a larger L or another tap")
    x_need = x_t + math.sqrt(r_t * (r_p - r_t))
    if not x_need > 0.0:
        raise ValueError(f"the {end} termination's reactance {x_t:.6g} ohm is too capacitive for a series tap capacitor to absorb")
    return 1.0 / (w0 * x_need)


def top_c_tap_reactive_f(n: float, f0: float, bw: float, l: float, r_term: float, x_term: float) -> float:
    """The capacitive end tap into a termination R + j X (F): the resistive tap of ``.c_tap`` with the termination's reactance absorbed."""
    order, g = _prototype(n, 1, RESONATOR_MAX_ORDER, "the resonator count n")
    fc, b = _narrowband(f0, bw)
    return _tap_reactive(g[0] * fc / b, 2.0 * math.pi * fc, _positive(l, "the resonator inductance"), _positive(r_term, "the termination's resistance"),
                         _finite(x_term, "the termination's reactance"), "end")


@dataclass(frozen=True)
class PortedTopC:
    """A top-C network designed between loaded ports: the source R_port shunted by L_p (Q_p), the load R_load shunted by more resistance.

    ``net`` is the Butterworth network designed between Re Z_source (with the
    source tap absorbing Im Z_source) and ``r_load_eff`` = R_load // the load
    port's other shunt resistance; ``g_extra`` = 1 / r_load_eff - 1 / R_load
    is that shunt as a conductance at the load node.
    """

    net: TopCNetwork
    r_port: float
    l_port: float
    q_port: float
    r_load: float
    g_extra: float


def top_c_ported_network(n: float, f0: float, bw: float, l: float, r_source: float, l_port: float, q_port: float, r_load: float,
                         r_load_eff: float) -> PortedTopC:
    """Design the top-C network of ``n`` resonators between a source port with a shunt inductor and a load port with extra shunt resistance."""
    z = port_shunt_l_impedance(r_source, l_port, q_port, f0)
    rl, re_ = _positive(r_load, "the load port's resistance"), _positive(r_load_eff, "the load the network sees")
    if re_ > rl * (1.0 + 1e-12):
        raise ValueError(f"the load the network sees ({re_:.6g} ohm) is the load port's {rl:.6g} ohm in parallel with more: it cannot exceed it")
    net = top_c_network(n, f0, bw, l, z.real, re_)
    c_s = _tap_reactive(net.q_e, 2.0 * math.pi * net.f0, net.l, z.real, z.imag, "source")
    return PortedTopC(net=replace(net, c_tap_source=c_s), r_port=float(r_source), l_port=float(l_port), q_port=float(q_port), r_load=rl,
                      g_extra=max(0.0, 1.0 / re_ - 1.0 / rl))


def top_c_ported_s21_complex(p: PortedTopC, q_u: float, f: float) -> complex:
    """The exact S21 of a ported top-C network at f: the source port's shunt inductor (series loss w0 L_p / Q_p fixed at f0), the network (every
    resonator inductor with w0 L / Q_u), the load port's extra shunt conductance; normalised to the two port resistances."""
    net = p.net
    w = _omega(f)
    r_q = 2.0 * math.pi * net.f0 * net.l / _positive(q_u, "the inductor's Q_u")
    r_qp = 2.0 * math.pi * net.f0 * p.l_port / p.q_port
    elements: list[tuple[str, complex]] = [("shunt", 1.0 / (r_qp + 1j * w * p.l_port)), ("series", _cap_z(w, net.c_tap_source))]
    for i in range(net.n):
        elements.append(("shunt", 1j * w * net.c_shunt[i] + 1.0 / (r_q + 1j * w * net.l)))
        if i < net.n - 1:
            elements.append(("series", _cap_z(w, net.c_couple[i])))
    elements.append(("series", _cap_z(w, net.c_tap_load)))
    if p.g_extra > 0.0:
        elements.append(("shunt", complex(p.g_extra, 0.0)))
    return _s21(elements, p.r_port, p.r_load, "calc.rf.resonator.top_c.ported_s21_db")


def top_c_ported_s21_db_value(n: float, f0: float, bw: float, l: float, r_source: float, l_port: float, q_port: float, r_load: float, r_load_eff: float,
                              q_u: float, f: float) -> float:
    """20 log10 |S21| (dB) of the ported top-C network at f."""
    p = top_c_ported_network(n, f0, bw, l, r_source, l_port, q_port, r_load, r_load_eff)
    return _db_mag(top_c_ported_s21_complex(p, q_u, f), "calc.rf.resonator.top_c.ported_s21_db")


def top_c_ported_rel_s21_db_value(n: float, f0: float, bw: float, l: float, r_source: float, l_port: float, q_port: float, r_load: float,
                                  r_load_eff: float, q_u: float, f: float, f_ref: float) -> float:
    """s21_db(f) - s21_db(f_ref) of the ported top-C network."""
    p = top_c_ported_network(n, f0, bw, l, r_source, l_port, q_port, r_load, r_load_eff)
    tool = "calc.rf.resonator.top_c.ported_rel_s21_db"
    return _db_mag(top_c_ported_s21_complex(p, q_u, f), tool) - _db_mag(top_c_ported_s21_complex(p, q_u, f_ref), tool)


def dissipation_loss_db(n: float, f0: float, bw: float, q_u: float) -> float:
    """Cohn's midband dissipation loss L0 = (10/ln 10) sum(g_i) f0 / (BW Q_u) of a Butterworth coupled-resonator filter (dB)."""
    order, g = _prototype(n, 1, RESONATOR_MAX_ORDER, "the resonator count n")
    fc, b = _narrowband(f0, bw)
    return DB_PER_NEPER * sum(g) * fc / (b * _positive(q_u, "the resonators' Q_u"))


def _detuning(q_l: float, f0: float, f: float) -> float:
    fc, fx = _positive(f0, "the resonant frequency"), _positive(f, "the frequency")
    return _positive(q_l, "the loaded Q") * (fx / fc - fc / fx)


def single_tuned_rejection_db(q_l: float, f0: float, f: float) -> float:
    """10 log10(1 + x^2), x = Q_L (f/f0 - f0/f): a single tuned circuit's rejection relative to resonance (symmetric ceiling)."""
    x = _detuning(q_l, f0, f)
    return 10.0 * math.log10(1.0 + x * x)


def double_tuned_rejection_db(q_l: float, f0: float, f: float) -> float:
    """10 log10(1 + x^4/4): a critically coupled pair's rejection relative to resonance (symmetric ceiling)."""
    x = _detuning(q_l, f0, f)
    x2 = x * x  # products, never a float **: it raises OverflowError where a product gives inf, refused as a sentence
    db = 10.0 * math.log10(1.0 + x2 * x2 / 4.0)
    if not math.isfinite(db):
        raise ValueError(f"calc.rf.resonator.double_tuned.rejection overflows: x = Q_L (f/f0 - f0/f) = {x:.6g} puts x^4/4 past the float range")
    return db


def single_tuned_insertion_loss_db(q_l: float, q_u: float) -> float:
    """-20 log10(1 - Q_L/Q_u): one resonator's loss when its total loaded Q_L includes its own Q_u (dB)."""
    ql, qu = _positive(q_l, "the loaded Q"), _positive(q_u, "the unloaded Q")
    if ql >= qu:
        raise ValueError(f"the loaded Q {ql:.6g} must stay below the unloaded Q {qu:.6g} (all the power would be lost in the resonator)")
    return -20.0 * math.log10(1.0 - ql / qu)


# --------------------------------------------------------------------------- crystal ladder (plain helpers)


@dataclass(frozen=True)
class CrystalLadder:
    """A C0-aware Butterworth crystal ladder (lower-sideband: series crystals, shunt coupling capacitors).

    ``f_mesh`` is the common mesh resonance (Hz), ``slope`` the meshes'
    reactance slope parameter x (ohm), ``x_crystal`` the crystal's reactance
    at f_mesh (ohm), ``c_couple`` the n - 1 shunt coupling capacitors,
    ``c_mesh`` each mesh's series tuning capacitor (``None`` for the mesh(es)
    with the largest coupling reactance, which set f_mesh), ``r_end`` the
    source and load resistance, ``l_m`` the motional inductance.
    """

    n: int
    bw: float
    f_s: float
    c_m: float
    c0: float
    l_m: float
    kappa: float
    f_mesh: float
    x_crystal: float
    slope: float
    c_couple: tuple[float, ...]
    c_mesh: tuple[float | None, ...]
    r_end: float


def ladder_k_values(n: float) -> list[float]:
    """The Butterworth coupling coefficients k_{i,i+1} = 1/sqrt(g_i g_{i+1}), i = 1 .. n-1."""
    order, g = _prototype(n, LADDER_MIN_ORDER, LADDER_MAX_ORDER, "the crystal count n")
    return [1.0 / math.sqrt(g[i] * g[i + 1]) for i in range(order - 1)]


def crystal_ladder(n: float, bw: float, f_s: float, c_m: float, c0: float) -> CrystalLadder:
    """Design the C0-aware Butterworth ladder of ``n`` crystals (series resonance f_s, motional C_m, holder C0) for BW."""
    order, g = _prototype(n, LADDER_MIN_ORDER, LADDER_MAX_ORDER, "the crystal count n")
    fs = _positive(f_s, "the crystals' series resonance")
    b = _positive(bw, "the bandwidth")
    cm = _positive(c_m, "the motional capacitance C_m")
    ch = _non_negative(c0, "the holder capacitance C0")
    if b >= fs:
        raise ValueError(f"a bandwidth of {b:.6g} Hz is not below the crystals' {fs:.6g} Hz")
    w_s = 2.0 * math.pi * fs
    lm = _nonzero_result(1.0 / (w_s * w_s * cm), "the crystals' motional inductance L_m = 1 / (w_s^2 C_m)")
    k = [1.0 / math.sqrt(g[i] * g[i + 1]) for i in range(order - 1)]
    sums = [(k[i - 1] if i > 0 else 0.0) + (k[i] if i < order - 1 else 0.0) for i in range(order)]
    kappa = max(sums)
    root = math.sqrt(4.0 * lm / cm)

    def omega(xm: float) -> float:  # the frequency above f_s where the motional reactance is xm
        return (xm + math.sqrt(xm * xm + root * root)) / (2.0 * lm)

    def state(xm: float) -> tuple[float, float, float, float]:
        w = omega(xm)
        d = 1.0 - w * ch * xm
        xc = xm / d
        slope = w / 2.0 * (lm + 1.0 / (w * w * cm) + ch * xm * xm) / (d * d) + xc / 2.0
        return w, d, xc, slope

    def numer(xm: float) -> float:  # (X_c - w_f kappa x) D^2: continuous up to the crystal's pole
        w = omega(xm)
        d = 1.0 - w * ch * xm
        frac = 2.0 * math.pi * b / w
        return xm * d - frac * kappa * (w / 2.0 * (lm + 1.0 / (w * w * cm) + ch * xm * xm) + xm * d / 2.0)

    if ch > 0.0:
        lo, hi = 0.0, 1.0
        while 1.0 - omega(hi) * ch * hi > 0.0:
            hi *= 2.0
        for _ in range(200):
            mid = (lo + hi) / 2.0
            if 1.0 - omega(mid) * ch * mid > 0.0:
                lo = mid
            else:
                hi = mid
        x_pole = lo
        a, c = 0.0, x_pole
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        p, q = c - gr * (c - a), a + gr * (c - a)
        for _ in range(200):
            if numer(p) > numer(q):
                c = q
            else:
                a = p
            p, q = c - gr * (c - a), a + gr * (c - a)
        x_top = (a + c) / 2.0
        if not numer(x_top) > 0.0:
            bw_max = fs * cm / (4.0 * ch * kappa)
            raise ValueError(
                f"no {order}-crystal Butterworth ladder of {b:.6g} Hz exists with C_m {cm:.6g} F and C0 {ch:.6g} F at {fs:.6g} Hz: "
                f"C0 limits this lower-sideband ladder to about f_s C_m / (4 C0 kappa) = {bw_max:.6g} Hz (kappa = {kappa:.6g}, "
                "the largest sum of a mesh's coupling coefficients) - a narrower bandwidth, fewer crystals or crystals with a larger C_m / C0 ratio")
        lo, hi = 0.0, x_top
    else:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            if numer(hi) > 0.0:
                break
            hi *= 2.0
        else:
            raise ValueError(f"no {order}-crystal ladder of {b:.6g} Hz exists at {fs:.6g} Hz: the coupling equation has no root")
    for _ in range(300):
        mid = (lo + hi) / 2.0
        if numer(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    xm = (lo + hi) / 2.0
    w0, _d, xc, slope = state(xm)
    frac = 2.0 * math.pi * b / w0
    kk = [frac * slope * ki for ki in k]
    cc = [1.0 / (w0 * ki) for ki in kk]
    mesh: list[float | None] = []
    for i in range(order):
        tune = xc - (kk[i - 1] if i > 0 else 0.0) - (kk[i] if i < order - 1 else 0.0)
        mesh.append(None if tune <= _MESH_TUNE_TOL * xc else 1.0 / (w0 * tune))
    return CrystalLadder(n=order, bw=b, f_s=fs, c_m=cm, c0=ch, l_m=lm, kappa=kappa, f_mesh=w0 / (2.0 * math.pi), x_crystal=xc, slope=slope,
                         c_couple=tuple(cc), c_mesh=tuple(mesh), r_end=frac * slope / g[0])


def crystal_ladder_s21(lad: CrystalLadder, r_m: float, f: float) -> complex:
    """The exact S21 of a designed ladder at f between R_end at both ends, each crystal with its motional R_m and C0."""
    w = _omega(f)
    rm = _non_negative(r_m, "the motional resistance R_m")
    z_m = rm + 1j * w * lad.l_m + _cap_z(w, lad.c_m)
    z_x = 0j if z_m == 0 else (z_m if lad.c0 == 0.0 else 1.0 / (1.0 / z_m + 1j * w * lad.c0))
    elements: list[tuple[str, complex]] = []
    for i in range(lad.n):
        z = z_x + (_cap_z(w, lad.c_mesh[i]) if lad.c_mesh[i] is not None else 0j)
        elements.append(("series", z))
        if i < lad.n - 1:
            elements.append(("shunt", 1j * w * lad.c_couple[i]))
    return _s21(elements, lad.r_end, lad.r_end, "calc.crystal.ladder.s21_db")


def crystal_ladder_s21_db_value(n: float, bw: float, f_s: float, c_m: float, c0: float, r_m: float, f: float) -> float:
    """20 log10 |S21| (dB) of the designed ladder at f (R_end terminations, motional R_m included)."""
    return _db_mag(crystal_ladder_s21(crystal_ladder(n, bw, f_s, c_m, c0), r_m, f), "calc.crystal.ladder.s21_db")


def crystal_ladder_center_hz(n: float, bw: float, f_s: float, c_m: float, c0: float, r_m: float) -> float:
    """The midpoint of the designed ladder's half-power edges (10 log10 2 dB below its peak), Hz."""
    lad = crystal_ladder(n, bw, f_s, c_m, c0)
    tool = "calc.crystal.ladder.center"

    def db(f: float) -> float:
        return _db_mag(crystal_ladder_s21(lad, r_m, f), tool)

    span = 1.5 * lad.bw
    steps = 600
    grid = [lad.f_mesh - span + 2.0 * span * i / steps for i in range(steps + 1)]
    vals = [db(f) for f in grid]
    i_pk = max(range(len(vals)), key=lambda i: vals[i])
    a = grid[max(i_pk - 1, 0)]
    c = grid[min(i_pk + 1, steps)]
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    p, q = c - gr * (c - a), a + gr * (c - a)
    for _ in range(100):
        if db(p) > db(q):
            c = q
        else:
            a = p
        p, q = c - gr * (c - a), a + gr * (c - a)
    f_pk = (a + c) / 2.0
    level = max(db(f_pk), vals[i_pk]) - HALF_POWER_DB
    step = lad.bw / 200.0

    def edge(direction: int) -> float:
        inside = f_pk
        outside = f_pk + direction * step
        for _ in range(20000):
            if outside <= 0.0:
                break
            if db(outside) < level:
                lo, hi = inside, outside
                for _ in range(100):
                    mid = (lo + hi) / 2.0
                    if db(mid) >= level:
                        lo = mid
                    else:
                        hi = mid
                return (lo + hi) / 2.0
            inside, outside = outside, outside + direction * step
        raise ValueError(f"{tool}: the designed ladder's response does not fall {HALF_POWER_DB:.4g} dB below its peak within {20000 * step:.6g} Hz")

    return (edge(-1) + edge(+1)) / 2.0


# --------------------------------------------------------------------------- pads, matches, detector, bias, levels (plain helpers)


def _pad_k(a_db: float) -> float:
    a = _positive(a_db, "the attenuation (dB)")
    return _pow10(a / 20.0, "the attenuation")


def pi_pad_r_shunt_ohm(z0: float, a_db: float) -> float:
    """The matched pi pad's shunt resistors Z0 (K + 1)/(K - 1), K = 10^(A/20)."""
    k = _pad_k(a_db)
    if k == 1.0:
        raise ValueError(f"an attenuation of {a_db!r} dB rounds to no attenuation: the shunt resistors would be infinite")
    return _nonzero_result(_positive(z0, "the pad's impedance Z0") * (k + 1.0) / (k - 1.0), "calc.rf.attenuator.pi.r_shunt")


def pi_pad_r_series_ohm(z0: float, a_db: float) -> float:
    """The matched pi pad's series resistor Z0 (K - 1/K)/2, K = 10^(A/20)."""
    k = _pad_k(a_db)
    return _nonzero_result(_positive(z0, "the pad's impedance Z0") * (k - 1.0 / k) / 2.0, "calc.rf.attenuator.pi.r_series")


def pa_load_line_ohm(v_cc: float, v_sat: float, p: float) -> float:
    """The PA's optimum load (V_CC - V_sat)^2 / (2 P) (Cripps' load line)."""
    vcc, vsat = _positive(v_cc, "the supply voltage"), _non_negative(v_sat, "the saturation voltage")
    if vsat >= vcc:
        raise ValueError(f"the saturation voltage {vsat:.6g} V leaves no swing below the supply {vcc:.6g} V")
    swing = vcc - vsat
    return _nonzero_result(swing * swing / (2.0 * _positive(p, "the output power")), "calc.rf.pa.load_line_r")


def quarter_wave_lumped_l_h(f: float, z0: float) -> float:
    """The lumped lambda/4 pi equivalent's series inductor Z0 / w."""
    return _nonzero_result(_positive(z0, "the line impedance Z0") / _omega(f), "calc.rf.quarter_wave_lumped.l")


def quarter_wave_lumped_c_f(f: float, z0: float) -> float:
    """The lumped lambda/4 pi equivalent's shunt capacitors 1 / (w Z0)."""
    return _nonzero_result(1.0 / (_omega(f) * _positive(z0, "the line impedance Z0")), "calc.rf.quarter_wave_lumped.c")


def quad_phase_deg(f: float, r_source: float, c_q: float, l: float, q_u: float, f_q: float, c_p: float, c_trim: float, r_p: float) -> float:
    """The exact phase of V(quad)/V_s: R_source and the coupling C_q into L (loss 2 pi f_q L/Q_u) // (C_p + C_trim) // R_p, degrees."""
    w = _omega(f)
    ind = _positive(l, "the quadrature inductance")
    r_l = _omega(f_q, "the frequency Q_u is stated at") * ind / _positive(q_u, "the inductor's Q_u")
    y = 1j * w * (_positive(c_p, "the tank capacitance") + _non_negative(c_trim, "the trimmer capacitance")) + 1.0 / _positive(r_p, "the tank resistance") + 1.0 / (1j * w * ind + r_l)
    z = _non_negative(r_source, "the source resistance") + _cap_z(w, _positive(c_q, "the quadrature capacitor"))
    h = _open_transfer([("series", z), ("shunt", y)], "calc.rf.quad.phase")
    return math.degrees(math.atan2(h.imag, h.real))


def bjt_bias_ic_a(v_b: float, v_be: float, r_e: float) -> float:
    """I_C ~ I_E = (V_B - V_BE) / R_E (beta -> infinity, V_B from an unloaded divider)."""
    vb, vbe = _finite(v_b, "the base voltage"), _positive(v_be, "V_BE")
    if vb <= vbe:
        raise ValueError(f"the base voltage {vb:.6g} V does not exceed V_BE {vbe:.6g} V: the transistor is off")
    return (vb - vbe) / _positive(r_e, "the emitter resistor")


def limiter_level_v(v_d: float, gain: float) -> float:
    """The clip level V_D x gain."""
    return _nonzero_result(_positive(v_d, "the diode drop") * _positive(gain, "the gain after the clipper"), "calc.rf.limiter.level")


def db_sum_value(a: float, b: float) -> float:
    """a + b (dB)."""
    return _finite(a, "the first level (dB)") + _finite(b, "the second level (dB)")


# --------------------------------------------------------------------------- audio (plain helpers)


def emphasis_corner_hz(tau: float) -> float:
    """The emphasis corner f_c = 1 / (2 pi tau)."""
    return _nonzero_result(1.0 / (2.0 * math.pi * _positive(tau, "the time constant")), "calc.audio.emphasis.corner")


def emphasis_db(f: float, f_c: float, sign: float) -> float:
    """sign x 10 log10(1 + (f/f_c)^2): +1 pre-emphasis (a rising response), -1 de-emphasis (dB)."""
    s = _sign(sign, "the emphasis sign (+1 pre-emphasis, -1 de-emphasis)")
    x = _non_negative(f, "the frequency") / _positive(f_c, "the corner frequency")
    return s * 10.0 * math.log10(1.0 + x * x)


def highpass1_db(f: float, f_c: float) -> float:
    """A first-order high-pass: -10 log10(1 + (f_c/f)^2) (dB)."""
    x = _positive(f_c, "the corner frequency") / _positive(f, "the frequency")
    return -10.0 * math.log10(1.0 + x * x)


def lowpass1_db(f: float, f_c: float) -> float:
    """A first-order low-pass: -10 log10(1 + (f/f_c)^2) (dB)."""
    x = _non_negative(f, "the frequency") / _positive(f_c, "the corner frequency")
    return -10.0 * math.log10(1.0 + x * x)


def lossy_integrator_db(f: float, tau_i: float, r_in: float, r_dc: float) -> float:
    """|H| of an integrator tau_i = R_in C with R_dc across C: (R_dc/R_in) / sqrt(1 + (w tau_i R_dc/R_in)^2), in dB."""
    w = _omega(f)
    ratio = _positive(r_dc, "the DC-limit resistor") / _positive(r_in, "the input resistor")
    x = w * _positive(tau_i, "the integrator time constant") * ratio
    return 20.0 * _log10_positive(ratio, "R_dc / R_in") - 10.0 * math.log10(1.0 + x * x)


def butterworth_pair_q(n: float, k: float) -> float:
    """The Butterworth k-th pole pair's Q_k = 1/g_k = 1/(2 sin((2k - 1) pi/(2n))), k = 1 .. floor(n/2)."""
    order = _integer(n, "the filter order n", 2, 20)
    index = _integer(k, "the pole-pair index k", 1, order // 2)
    return 1.0 / (2.0 * math.sin((2 * index - 1) * math.pi / (2 * order)))


def _sallen_key(q: float, f_c: float, r: float) -> tuple[float, float]:
    qq = _positive(q, "the section Q")
    wr = _omega(f_c, "the corner frequency") * _positive(r, "the resistors R")
    return 2.0 * qq / wr, 1.0 / (2.0 * qq * wr)


def sallen_key_c1_f(q: float, f_c: float, r: float) -> float:
    """Unity-gain Sallen-Key low-pass, R1 = R2 = R: the feedback capacitor C1 = 2 Q / (w_c R)."""
    return _nonzero_result(_sallen_key(q, f_c, r)[0], "calc.audio.sallen_key.c1")


def sallen_key_c2_f(q: float, f_c: float, r: float) -> float:
    """Unity-gain Sallen-Key low-pass, R1 = R2 = R: the capacitor to ground C2 = 1 / (2 Q w_c R)."""
    return _nonzero_result(_sallen_key(q, f_c, r)[1], "calc.audio.sallen_key.c2")


def tot_period_s(k_rc: float, r: float, c: float) -> float:
    """The 4060 time-out: 2^13 oscillator periods k_RC R C until Q14 first rises."""
    period = _positive(k_rc, "the RC-oscillator constant k_RC") * _positive(r, "the timing resistor") * _positive(c, "the timing capacitor")
    return _nonzero_result(TOT_Q14_PERIODS * period, "calc.rf.tot.period")


def rail_budget_a(i_load: float, i_rating: float) -> float:
    """The regulator's rating margin I_rating - I_load (A; negative when the load exceeds the rating)."""
    return _positive(i_rating, "the regulator's current rating") - _non_negative(i_load, "the rail's load current")


def regulator_headroom_v(v_in_min: float, v_out: float, v_dropout: float, i_load: float, r_path: float) -> float:
    """V_in,min - I_load R_path - V_dropout - V_out (V; negative: out of regulation at the minimum input)."""
    return (_positive(v_in_min, "the minimum input voltage") - _non_negative(i_load, "the load current") * _non_negative(r_path, "the path resistance")
            - _non_negative(v_dropout, "the dropout voltage") - _positive(v_out, "the output voltage"))


# --------------------------------------------------------------------------- traced calculators (registered)


def mult_stage(f: Traced[float], m: Traced[float], ids: tuple[str, str] = ("f", "m")) -> Traced[float]:
    """A multiplier stage's output M f (Hz)."""
    return _derived(multiplier_stage_hz(_num(f, "f"), _num(m, "m")), "calc.rf.mult.stage", ids, "Hz", "f_out = M f_in")


def mult_spur(f_out: Traced[float], f_x: Traced[float], k: Traced[float], ids: tuple[str, str, str] = ("f_out", "f_x", "k")) -> Traced[float]:
    """A close-in multiplier product f_out + k f_x (Hz)."""
    return _derived(multiplier_spur_hz(_num(f_out, "f_out"), _num(f_x, "f_x"), _num(k, "k")), "calc.rf.mult.spur", ids, "Hz", "f = f_out + k f_x")


def harmonic(f: Traced[float], k: Traced[float], ids: tuple[str, str] = ("f", "k")) -> Traced[float]:
    """The k-th harmonic k f (Hz)."""
    return _derived(harmonic_hz(_num(f, "f"), _num(k, "k")), "calc.rf.harmonic", ids, "Hz", "f_k = k f")


def superhet_lo(f: Traced[float], f_if: Traced[float], side: Traced[float], ids: tuple[str, str, str] = ("f", "f_if", "side")) -> Traced[float]:
    """LO = f + side IF (Hz; side -1 low, +1 high)."""
    return _derived(superhet_lo_hz(_num(f, "f"), _num(f_if, "f_if"), _num(side, "side")), "calc.rf.superhet.lo", ids, "Hz", "LO = f + side IF (side -1: low-side LO, +1: high side)")


def superhet_image(f: Traced[float], lo: Traced[float], ids: tuple[str, str] = ("f", "lo")) -> Traced[float]:
    """The image 2 LO - f (Hz)."""
    return _derived(superhet_image_hz(_num(f, "f"), _num(lo, "lo")), "calc.rf.superhet.image", ids, "Hz", "f_image = 2 LO - f (LO - IF for a low-side LO)")


def superhet_half_if(f: Traced[float], lo: Traced[float], ids: tuple[str, str] = ("f", "lo")) -> Traced[float]:
    """The half-IF response (f + LO)/2 (Hz)."""
    return _derived(superhet_half_if_hz(_num(f, "f"), _num(lo, "lo")), "calc.rf.superhet.half_if", ids, "Hz", "f_half_IF = (f + LO) / 2 (LO + IF/2 for a low-side LO)")


def superhet_second_image(f: Traced[float], f_if2: Traced[float], ids: tuple[str, str] = ("f", "f_if2")) -> Traced[float]:
    """The second image f - 2 IF2 at the antenna (Hz; low-side first and second LO)."""
    return _derived(superhet_second_image_hz(_num(f, "f"), _num(f_if2, "f_if2")), "calc.rf.superhet.second_image", ids, "Hz", "f_2nd_image = f - 2 IF2 (low-side LO1 and LO2)")


def superhet_lo_spur_response(lo: Traced[float], f_r: Traced[float], k: Traced[float], f_if: Traced[float], if_sign: Traced[float],
                              ids: tuple[str, str, str, str, str] = ("lo", "f_r", "k", "f_if", "if_sign")) -> Traced[float]:
    """The response of an LO spur at LO + k f_R: LO + k f_R + s IF (Hz)."""
    value = superhet_lo_spur_response_hz(_num(lo, "lo"), _num(f_r, "f_r"), _num(k, "k"), _num(f_if, "f_if"), _num(if_sign, "if_sign"))
    return _derived(value, "calc.rf.superhet.lo_spur_response", ids, "Hz", "f = LO + k f_R + s IF (s = -1 or +1)")


def fm_obw99(delta_f: Traced[float], f_m: Traced[float], ids: tuple[str, str] = ("delta_f", "f_m")) -> Traced[float]:
    """The 99 % occupied bandwidth of single-tone FM (Hz)."""
    return _derived(fm_obw99_hz(_num(delta_f, "delta_f"), _num(f_m, "f_m")), "calc.rf.fm.obw99", ids, "Hz",
                    "B99 = smallest 2 n f_m with J0(beta)^2 + 2 sum_(k=1..n) J_k(beta)^2 >= 0.99, beta = delta_f / f_m")


def fm_pm_integrator_tau(n_mult: Traced[float], k_pm: Traced[float], v_lim: Traced[float], delta_f: Traced[float],
                         ids: tuple[str, str, str, str] = ("n_mult", "k_pm", "v_lim", "delta_f")) -> Traced[float]:
    """The PM-to-FM integrator's time constant N K_pm V_lim / (2 pi delta_f) (s)."""
    value = pm_integrator_tau_s(_num(n_mult, "n_mult"), _num(k_pm, "k_pm"), _num(v_lim, "v_lim"), _num(delta_f, "delta_f"))
    return _derived(value, "calc.rf.fm.pm_integrator_tau", ids, "s", "tau_i = N K_pm V_lim / (2 pi delta_f)")


def fm_pm_drive_limit(n_mult: Traced[float], k_pm: Traced[float], tau_i: Traced[float], delta_f: Traced[float],
                      ids: tuple[str, str, str, str] = ("n_mult", "k_pm", "tau_i", "delta_f")) -> Traced[float]:
    """The largest integrator drive 2 pi tau_i delta_f / (N K_pm) (V)."""
    value = pm_drive_limit_v(_num(n_mult, "n_mult"), _num(k_pm, "k_pm"), _num(tau_i, "tau_i"), _num(delta_f, "delta_f"))
    return _derived(value, "calc.rf.fm.pm_drive_limit", ids, "V", "V_max = 2 pi tau_i delta_f / (N K_pm)")


def varactor_c_at_bias(cjo: Traced[float], vj: Traced[float], m: Traced[float], v: Traced[float], ids: tuple[str, str, str, str] = ("cjo", "vj", "m", "v")) -> Traced[float]:
    """A varactor's capacitance CJO / (1 + V/VJ)^M at reverse bias V (F)."""
    return _derived(varactor_c_f(_num(cjo, "cjo"), _num(vj, "vj"), _num(m, "m"), _num(v, "v")), "calc.rf.varactor.c_at_bias", ids, "F", "C = CJO / (1 + V / VJ)^M (SPICE junction law, reverse bias V >= 0)")


def varactor_dc_dv(cjo: Traced[float], vj: Traced[float], m: Traced[float], v: Traced[float], ids: tuple[str, str, str, str] = ("cjo", "vj", "m", "v")) -> Traced[float]:
    """A varactor's slope -M C / (VJ + V) (F/V)."""
    return _derived(varactor_slope_f_per_v(_num(cjo, "cjo"), _num(vj, "vj"), _num(m, "m"), _num(v, "v")), "calc.rf.varactor.dc_dv", ids, "F/V", "dC/dV = -M C / (VJ + V), C = CJO / (1 + V / VJ)^M")


def pm_k_pm(q_l: Traced[float], dc_dv: Traced[float], c_tot: Traced[float], ids: tuple[str, str, str] = ("q_l", "dc_dv", "c_tot")) -> Traced[float]:
    """One tank's small-signal PM constant -Q_L (dC/dV) / C_tot (rad/V)."""
    return _derived(pm_k_pm_rad_per_v(_num(q_l, "q_l"), _num(dc_dv, "dc_dv"), _num(c_tot, "c_tot")), "calc.rf.pm.k_pm", ids, "rad/V", "K_pm = -Q_L (dC/dV) / C_tot (per tank, at resonance)")


def pm_c_fixed(c_tot: Traced[float], c_var: Traced[float], c_trim: Traced[float], ids: tuple[str, str, str] = ("c_tot", "c_var", "c_trim")) -> Traced[float]:
    """The PM tank's fixed capacitor C_tot - C_var - C_trim (F)."""
    return _derived(pm_c_fixed_f(_num(c_tot, "c_tot"), _num(c_var, "c_var"), _num(c_trim, "c_trim")), "calc.rf.pm.c_fixed", ids, "F", "C_fixed = C_tot - C_var(V0) - C_trim")


def pm_source_r(f: Traced[float], l: Traced[float], q_l: Traced[float], q_u: Traced[float], r_port: Traced[float],
                ids: tuple[str, str, str, str, str] = ("f", "l", "q_l", "q_u", "r_port")) -> Traced[float]:
    """The PM tank's series source resistance for the loaded Q (ohm)."""
    value = pm_source_r_ohm(_num(f, "f"), _num(l, "l"), _num(q_l, "q_l"), _num(q_u, "q_u"), _num(r_port, "r_port"))
    return _derived(value, "calc.rf.pm.source_r", ids, "ohm", "R_s = 1 / (1 / (Q_L w L) - 1 / (Q_u w L)) - R_port (DC block neglected)")


def pm_source_r_loaded(f: Traced[float], l: Traced[float], q_l: Traced[float], q_u: Traced[float], r_port: Traced[float], r_load: Traced[float],
                       ids: tuple[str, str, str, str, str, str] = ("f", "l", "q_l", "q_u", "r_port", "r_load")) -> Traced[float]:
    """The PM tank's series source resistance for the loaded Q with a load R_load at the tank (ohm)."""
    value = pm_source_r_loaded_ohm(_num(f, "f"), _num(l, "l"), _num(q_l, "q_l"), _num(q_u, "q_u"), _num(r_port, "r_port"), _num(r_load, "r_load"))
    return _derived(value, "calc.rf.pm.source_r_loaded", ids, "ohm", "R_s = 1 / (1 / (Q_L w L) - 1 / (Q_u w L) - 1 / R_load) - R_port (DC block neglected)")


_TANK_NOTE = ("phi = arg V(tank) / V_s of R_port - C_dc - R_s into (C_fixed + C_trim + C_var) // (L + 2 pi f_q L / Q_u + (C_bypass // R_feed))"
              " (exact network, not the ideal -atan(Q (f/f0 - f0/f)))")


def pm_tank_phase(f: Traced[float], f_q: Traced[float], l: Traced[float], q_u: Traced[float], c_fixed: Traced[float], c_trim: Traced[float],
                  c_var: Traced[float], r_port: Traced[float], c_dc: Traced[float], r_s: Traced[float], c_bypass: Traced[float], r_feed: Traced[float],
                  ids: tuple[str, ...] = ("f", "f_q", "l", "q_u", "c_fixed", "c_trim", "c_var", "r_port", "c_dc", "r_s", "c_bypass", "r_feed")) -> Traced[float]:
    """The exact phase of the designed PM tank at f, the tank node probed (deg)."""
    value = pm_tank_phase_deg(_num(f, "f"), _num(f_q, "f_q"), _num(l, "l"), _num(q_u, "q_u"), _num(c_fixed, "c_fixed"), _num(c_trim, "c_trim"),
                              _num(c_var, "c_var"), _num(r_port, "r_port"), _num(c_dc, "c_dc"), _num(r_s, "r_s"), _num(c_bypass, "c_bypass"), _num(r_feed, "r_feed"))
    return _derived(value, "calc.rf.pm.tank_phase", ids, "deg", _TANK_NOTE)


def pm_tank_phase_loaded(f: Traced[float], f_q: Traced[float], l: Traced[float], q_u: Traced[float], c_fixed: Traced[float], c_trim: Traced[float],
                         c_var: Traced[float], r_port: Traced[float], c_dc: Traced[float], r_s: Traced[float], c_bypass: Traced[float], r_feed: Traced[float],
                         r_load: Traced[float],
                         ids: tuple[str, ...] = ("f", "f_q", "l", "q_u", "c_fixed", "c_trim", "c_var", "r_port", "c_dc", "r_s", "c_bypass", "r_feed", "r_load")) -> Traced[float]:
    """The exact phase of the designed PM tank at f with a load R_load at the tank node (deg)."""
    value = pm_tank_phase_loaded_deg(_num(f, "f"), _num(f_q, "f_q"), _num(l, "l"), _num(q_u, "q_u"), _num(c_fixed, "c_fixed"), _num(c_trim, "c_trim"),
                                     _num(c_var, "c_var"), _num(r_port, "r_port"), _num(c_dc, "c_dc"), _num(r_s, "r_s"), _num(c_bypass, "c_bypass"),
                                     _num(r_feed, "r_feed"), _num(r_load, "r_load"))
    return _derived(value, "calc.rf.pm.tank_phase_loaded", ids, "deg", _TANK_NOTE + "; R_load across the tank")


def q_parallel(f: Traced[float], l: Traced[float], r: Traced[float], ids: tuple[str, str, str] = ("f", "l", "r")) -> Traced[float]:
    """The Q of a parallel R across L: R / (w L)."""
    return _derived(q_parallel_value(_num(f, "f"), _num(l, "l"), _num(r, "r")), "calc.rf.q_parallel", ids, None, "Q = R / (2 pi f L)")


def top_c_bw_for_qe(n: Traced[float], f0: Traced[float], q_e: Traced[float], ids: tuple[str, str, str] = ("n", "f0", "q_e")) -> Traced[float]:
    """The top-C network bandwidth g_1 f0 / Q_e for an end external Q (Hz)."""
    return _derived(top_c_bw_for_qe_hz(_num(n, "n"), _num(f0, "f0"), _num(q_e, "q_e")), "calc.rf.resonator.top_c.bw_for_qe", ids, "Hz", "BW = g_1 f0 / Q_e (Butterworth g)")


def top_c_c_couple(n: Traced[float], i: Traced[float], f0: Traced[float], bw: Traced[float], l: Traced[float],
                   ids: tuple[str, str, str, str, str] = ("n", "i", "f0", "bw", "l")) -> Traced[float]:
    """The top coupling capacitor between resonators i and i+1: k_i (BW/f0) C_res (F)."""
    order = _integer(_num(n, "n"), "the resonator count n", 1, RESONATOR_MAX_ORDER)
    if order < 2:
        raise ValueError("a single resonator has no coupling capacitor")
    fc, b = _narrowband(_num(f0, "f0"), _num(bw, "bw"))
    index = _integer(_num(i, "i"), "the coupling index i", 1, order - 1)
    g = butterworth_g_values(order)
    w0 = 2.0 * math.pi * fc
    c_res = 1.0 / (w0 * w0 * _positive(_num(l, "l"), "the resonator inductance"))
    value = c_res * (b / fc) / math.sqrt(g[index - 1] * g[index])
    return _derived(_nonzero_result(value, "calc.rf.resonator.top_c.c_couple"), "calc.rf.resonator.top_c.c_couple", ids, "F",
                    "C_k = k_i (BW / f0) C_res, k_i = 1 / sqrt(g_i g_(i+1)), C_res = 1 / (w0^2 L) (Dishal; Butterworth g)")


def top_c_c_tap(n: Traced[float], f0: Traced[float], bw: Traced[float], l: Traced[float], r_term: Traced[float],
                ids: tuple[str, str, str, str, str] = ("n", "f0", "bw", "l", "r_term")) -> Traced[float]:
    """The capacitive end tap into a termination R_t (F)."""
    order, g = _prototype(_num(n, "n"), 1, RESONATOR_MAX_ORDER, "the resonator count n")
    fc, b = _narrowband(_num(f0, "f0"), _num(bw, "bw"))
    w0 = 2.0 * math.pi * fc
    _rp, c_s, _ceq = _tap(g[0] * fc / b, w0, _positive(_num(l, "l"), "the resonator inductance"), _positive(_num(r_term, "r_term"), "the termination"), "end")
    return _derived(c_s, "calc.rf.resonator.top_c.c_tap", ids, "F", "C_s = 1 / (w0 sqrt(R_p R_t - R_t^2)), R_p = Q_e w0 L, Q_e = g_1 f0 / BW")


def top_c_c_shunt(n: Traced[float], i: Traced[float], f0: Traced[float], bw: Traced[float], l: Traced[float], r_source: Traced[float], r_load: Traced[float],
                  ids: tuple[str, ...] = ("n", "i", "f0", "bw", "l", "r_source", "r_load")) -> Traced[float]:
    """Resonator i's shunt capacitor after the couplings and the taps' parallel equivalents (F)."""
    net = top_c_network(_num(n, "n"), _num(f0, "f0"), _num(bw, "bw"), _num(l, "l"), _num(r_source, "r_source"), _num(r_load, "r_load"))
    index = _integer(_num(i, "i"), "the resonator index i", 1, net.n)
    return _derived(net.c_shunt[index - 1], "calc.rf.resonator.top_c.c_shunt", ids, "F",
                    "C_i = C_res - C_k(i-1) - C_k(i) - C_s / (1 + (w0 C_s R_t)^2) at an end")


def top_c_s21_db(n: Traced[float], f0: Traced[float], bw: Traced[float], l: Traced[float], r_source: Traced[float], r_load: Traced[float],
                 q_u: Traced[float], f: Traced[float], ids: tuple[str, ...] = ("n", "f0", "bw", "l", "r_source", "r_load", "q_u", "f")) -> Traced[float]:
    """The exact S21 of the designed top-C network at f (dB)."""
    value = top_c_s21_db_value(_num(n, "n"), _num(f0, "f0"), _num(bw, "bw"), _num(l, "l"), _num(r_source, "r_source"), _num(r_load, "r_load"), _num(q_u, "q_u"), _num(f, "f"))
    return _derived(value, "calc.rf.resonator.top_c.s21_db", ids, "dB", "S21 = 20 log10 |2 V_out / V_s sqrt(R_s / R_l)| of the designed top-C network (inductor loss w0 L / Q_u; exact)")


def top_c_rel_s21_db(n: Traced[float], f0: Traced[float], bw: Traced[float], l: Traced[float], r_source: Traced[float], r_load: Traced[float],
                     q_u: Traced[float], f: Traced[float], f_ref: Traced[float],
                     ids: tuple[str, ...] = ("n", "f0", "bw", "l", "r_source", "r_load", "q_u", "f", "f_ref")) -> Traced[float]:
    """S21(f) - S21(f_ref) of the designed top-C network (dB)."""
    value = top_c_rel_s21_db_value(_num(n, "n"), _num(f0, "f0"), _num(bw, "bw"), _num(l, "l"), _num(r_source, "r_source"), _num(r_load, "r_load"),
                                   _num(q_u, "q_u"), _num(f, "f"), _num(f_ref, "f_ref"))
    return _derived(value, "calc.rf.resonator.top_c.rel_s21_db", ids, "dB", "rel = S21_dB(f) - S21_dB(f_ref) of the designed top-C network (exact)")


_PORT_IDS = ("r_port", "l_port", "q_port", "f0")
_PORTED_IDS = ("n", "f0", "bw", "l", "r_source", "l_port", "q_port", "r_load", "r_load_eff", "q_u", "f")


def top_c_port_r(r_port: Traced[float], l_port: Traced[float], q_port: Traced[float], f0: Traced[float], ids: tuple[str, ...] = _PORT_IDS) -> Traced[float]:
    """Re (R_port // (L_p + w0 L_p / Q_p)) at f0 (ohm)."""
    value = top_c_port_r_ohm(_num(r_port, "r_port"), _num(l_port, "l_port"), _num(q_port, "q_port"), _num(f0, "f0"))
    return _derived(value, "calc.rf.resonator.top_c.port_r", ids, "ohm", "Re Z, Z = R_port (j w0 L_p + w0 L_p / Q_p) / (R_port + j w0 L_p + w0 L_p / Q_p)")


def top_c_port_x(r_port: Traced[float], l_port: Traced[float], q_port: Traced[float], f0: Traced[float], ids: tuple[str, ...] = _PORT_IDS) -> Traced[float]:
    """Im (R_port // (L_p + w0 L_p / Q_p)) at f0 (ohm)."""
    value = top_c_port_x_ohm(_num(r_port, "r_port"), _num(l_port, "l_port"), _num(q_port, "q_port"), _num(f0, "f0"))
    return _derived(value, "calc.rf.resonator.top_c.port_x", ids, "ohm", "Im Z, Z = R_port (j w0 L_p + w0 L_p / Q_p) / (R_port + j w0 L_p + w0 L_p / Q_p)")


def top_c_c_tap_reactive(n: Traced[float], f0: Traced[float], bw: Traced[float], l: Traced[float], r_term: Traced[float], x_term: Traced[float],
                         ids: tuple[str, ...] = ("n", "f0", "bw", "l", "r_term", "x_term")) -> Traced[float]:
    """The capacitive end tap into a termination R + j X (F)."""
    value = top_c_tap_reactive_f(_num(n, "n"), _num(f0, "f0"), _num(bw, "bw"), _num(l, "l"), _num(r_term, "r_term"), _num(x_term, "x_term"))
    return _derived(value, "calc.rf.resonator.top_c.c_tap_reactive", ids, "F",
                    "C_s = 1 / (w0 (X_t + sqrt(R_t (R_p - R_t)))), R_p = Q_e w0 L, Q_e = g_1 f0 / BW (the resistive tap with X_t absorbed)")


def _ported_args(vals: tuple[Traced[float], ...]) -> tuple[float, ...]:
    return tuple(_num(v, name) for v, name in zip(vals, _PORTED_IDS))


def top_c_ported_s21_db(n: Traced[float], f0: Traced[float], bw: Traced[float], l: Traced[float], r_source: Traced[float], l_port: Traced[float],
                        q_port: Traced[float], r_load: Traced[float], r_load_eff: Traced[float], q_u: Traced[float], f: Traced[float],
                        ids: tuple[str, ...] = _PORTED_IDS) -> Traced[float]:
    """The exact S21 of the ported top-C network at f (dB)."""
    value = top_c_ported_s21_db_value(*_ported_args((n, f0, bw, l, r_source, l_port, q_port, r_load, r_load_eff, q_u, f)))
    return _derived(value, "calc.rf.resonator.top_c.ported_s21_db", ids, "dB",
                    "S21 = 20 log10 |2 V_out / V_s sqrt(R_source / R_load)| of the top-C network with the source port's shunt L_p (loss w0 L_p / Q_p) and "
                    "the load port's extra shunt R (1 / (1 / r_load_eff - 1 / R_load)); taps designed for Re Z_source (Im Z absorbed) and r_load_eff (exact)")


def top_c_ported_rel_s21_db(n: Traced[float], f0: Traced[float], bw: Traced[float], l: Traced[float], r_source: Traced[float], l_port: Traced[float],
                            q_port: Traced[float], r_load: Traced[float], r_load_eff: Traced[float], q_u: Traced[float], f: Traced[float],
                            f_ref: Traced[float], ids: tuple[str, ...] = (*_PORTED_IDS, "f_ref")) -> Traced[float]:
    """S21(f) - S21(f_ref) of the ported top-C network (dB)."""
    value = top_c_ported_rel_s21_db_value(*_ported_args((n, f0, bw, l, r_source, l_port, q_port, r_load, r_load_eff, q_u, f)), _num(f_ref, "f_ref"))
    return _derived(value, "calc.rf.resonator.top_c.ported_rel_s21_db", ids, "dB", "rel = S21_dB(f) - S21_dB(f_ref) of the ported top-C network (exact)")


def bpf_dissipation_loss(n: Traced[float], f0: Traced[float], bw: Traced[float], q_u: Traced[float], ids: tuple[str, str, str, str] = ("n", "f0", "bw", "q_u")) -> Traced[float]:
    """Cohn's midband dissipation loss of a Butterworth coupled-resonator filter (dB)."""
    return _derived(dissipation_loss_db(_num(n, "n"), _num(f0, "f0"), _num(bw, "bw"), _num(q_u, "q_u")), "calc.rf.bpf.dissipation_loss", ids, "dB",
                    "L0 = 4.343 sum(g_i) f0 / (BW Q_u) (Cohn 1959; Butterworth g)")


def single_tuned_rejection(q_l: Traced[float], f0: Traced[float], f: Traced[float], ids: tuple[str, str, str] = ("q_l", "f0", "f")) -> Traced[float]:
    """A single tuned circuit's symmetric rejection 10 log10(1 + x^2) (dB)."""
    return _derived(single_tuned_rejection_db(_num(q_l, "q_l"), _num(f0, "f0"), _num(f, "f")), "calc.rf.resonator.single_tuned.rejection", ids, "dB",
                    "A = 10 log10(1 + x^2), x = Q_L (f / f0 - f0 / f) (symmetric ceiling)")


def single_tuned_insertion_loss(q_l: Traced[float], q_u: Traced[float], ids: tuple[str, str] = ("q_l", "q_u")) -> Traced[float]:
    """One resonator's loss -20 log10(1 - Q_L/Q_u) (dB)."""
    return _derived(single_tuned_insertion_loss_db(_num(q_l, "q_l"), _num(q_u, "q_u")), "calc.rf.resonator.single_tuned.insertion_loss", ids, "dB", "IL = -20 log10(1 - Q_L / Q_u)")


def double_tuned_rejection(q_l: Traced[float], f0: Traced[float], f: Traced[float], ids: tuple[str, str, str] = ("q_l", "f0", "f")) -> Traced[float]:
    """A critically coupled pair's symmetric rejection 10 log10(1 + x^4/4) (dB)."""
    return _derived(double_tuned_rejection_db(_num(q_l, "q_l"), _num(f0, "f0"), _num(f, "f")), "calc.rf.resonator.double_tuned.rejection", ids, "dB",
                    "A = 10 log10(1 + x^4 / 4), x = Q_L (f / f0 - f0 / f) (critical coupling; symmetric ceiling)")


def ladder_k(n: Traced[float], i: Traced[float], ids: tuple[str, str] = ("n", "i")) -> Traced[float]:
    """The ladder's Butterworth coupling coefficient k_{i,i+1}."""
    k = ladder_k_values(_num(n, "n"))
    index = _integer(_num(i, "i"), "the coupling index i", 1, len(k))
    return _derived(k[index - 1], "calc.crystal.ladder.k", ids, None, "k_(i,i+1) = 1 / sqrt(g_i g_(i+1)) (Butterworth)")


def ladder_q(n: Traced[float], ids: tuple[str] = ("n",)) -> Traced[float]:
    """The ladder's Butterworth end q = g_1."""
    order, g = _prototype(_num(n, "n"), LADDER_MIN_ORDER, LADDER_MAX_ORDER, "the crystal count n")
    return _derived(g[0], "calc.crystal.ladder.q", ids, None, "q = g_1 (Butterworth)")


_LADDER_NOTE = "C0-aware Dishal ladder (reactance slope x of crystal + C0 at the mesh frequency; MYJ 8.02, EMRFD ch. 3)"


def _ladder_from(n: Traced, bw: Traced, f_s: Traced, c_m: Traced, c0: Traced) -> CrystalLadder:
    return crystal_ladder(_num(n, "n"), _num(bw, "bw"), _num(f_s, "f_s"), _num(c_m, "c_m"), _num(c0, "c0"))


def ladder_center(n: Traced[float], bw: Traced[float], f_s: Traced[float], c_m: Traced[float], c0: Traced[float], r_m: Traced[float],
                  ids: tuple[str, ...] = ("n", "bw", "f_s", "c_m", "c0", "r_m")) -> Traced[float]:
    """The designed ladder's passband centre: the midpoint of its half-power edges (Hz)."""
    value = crystal_ladder_center_hz(_num(n, "n"), _num(bw, "bw"), _num(f_s, "f_s"), _num(c_m, "c_m"), _num(c0, "c0"), _num(r_m, "r_m"))
    return _derived(value, "calc.crystal.ladder.center", ids, "Hz", f"f_0 = (f_-3dB,lo + f_-3dB,hi) / 2 of the designed ladder's exact response; {_LADDER_NOTE}")


def ladder_c_couple(n: Traced[float], i: Traced[float], bw: Traced[float], f_s: Traced[float], c_m: Traced[float], c0: Traced[float],
                    ids: tuple[str, ...] = ("n", "i", "bw", "f_s", "c_m", "c0")) -> Traced[float]:
    """The shunt coupling capacitor between crystals i and i+1 (F)."""
    lad = _ladder_from(n, bw, f_s, c_m, c0)
    index = _integer(_num(i, "i"), "the coupling index i", 1, lad.n - 1)
    return _derived(lad.c_couple[index - 1], "calc.crystal.ladder.c_couple", ids, "F",
                    f"C_(i,i+1) = 1 / (w0 K), K = (BW / f0) x k_(i,i+1) (-> C_m f_0 / (k BW) for C0 = 0); {_LADDER_NOTE}")


def ladder_r_end(n: Traced[float], bw: Traced[float], f_s: Traced[float], c_m: Traced[float], c0: Traced[float],
                 ids: tuple[str, ...] = ("n", "bw", "f_s", "c_m", "c0")) -> Traced[float]:
    """The ladder's source and load resistance (ohm)."""
    lad = _ladder_from(n, bw, f_s, c_m, c0)
    return _derived(lad.r_end, "calc.crystal.ladder.r_end", ids, "ohm", f"R_end = (BW / f0) x / q (-> 2 pi BW L_m / q for C0 = 0); {_LADDER_NOTE}")


def ladder_mesh_c(n: Traced[float], i: Traced[float], bw: Traced[float], f_s: Traced[float], c_m: Traced[float], c0: Traced[float],
                  ids: tuple[str, ...] = ("n", "i", "bw", "f_s", "c_m", "c0")) -> Traced[float]:
    """Mesh i's series tuning capacitor (F); refused for a mesh that sets the mesh frequency and needs none."""
    lad = _ladder_from(n, bw, f_s, c_m, c0)
    index = _integer(_num(i, "i"), "the mesh index i", 1, lad.n)
    c = lad.c_mesh[index - 1]
    if c is None:
        raise ValueError(f"mesh {index} carries the largest coupling reactance and sets the mesh frequency: it needs no tuning capacitor")
    return _derived(c, "calc.crystal.ladder.mesh_c", ids, "F", f"C_t,i = 1 / (w0 (X_c - K_(i-1,i) - K_(i,i+1))); {_LADDER_NOTE}")


def ladder_s21_db(n: Traced[float], bw: Traced[float], f_s: Traced[float], c_m: Traced[float], c0: Traced[float], r_m: Traced[float], f: Traced[float],
                  ids: tuple[str, ...] = ("n", "bw", "f_s", "c_m", "c0", "r_m", "f")) -> Traced[float]:
    """The designed ladder's exact S21 at f between R_end terminations (dB)."""
    value = crystal_ladder_s21_db_value(_num(n, "n"), _num(bw, "bw"), _num(f_s, "f_s"), _num(c_m, "c_m"), _num(c0, "c0"), _num(r_m, "r_m"), _num(f, "f"))
    return _derived(value, "calc.crystal.ladder.s21_db", ids, "dB", f"S21 = 20 log10 |2 V_out / V_s| between R_end, crystals with R_m and C0 (exact); {_LADDER_NOTE}")


def attenuator_pi_r_shunt(z0: Traced[float], a: Traced[float], ids: tuple[str, str] = ("z0", "a")) -> Traced[float]:
    """The matched pi pad's shunt resistors (ohm)."""
    return _derived(pi_pad_r_shunt_ohm(_num(z0, "z0"), _num(a, "a")), "calc.rf.attenuator.pi.r_shunt", ids, "ohm", "R_sh = Z0 (K + 1) / (K - 1), K = 10^(A / 20)")


def attenuator_pi_r_series(z0: Traced[float], a: Traced[float], ids: tuple[str, str] = ("z0", "a")) -> Traced[float]:
    """The matched pi pad's series resistor (ohm)."""
    return _derived(pi_pad_r_series_ohm(_num(z0, "z0"), _num(a, "a")), "calc.rf.attenuator.pi.r_series", ids, "ohm", "R_se = Z0 (K - 1 / K) / 2, K = 10^(A / 20)")


def pa_load_line_r(v_cc: Traced[float], v_sat: Traced[float], p: Traced[float], ids: tuple[str, str, str] = ("v_cc", "v_sat", "p")) -> Traced[float]:
    """The PA's load-line resistance (V_CC - V_sat)^2 / (2 P) (ohm)."""
    return _derived(pa_load_line_ohm(_num(v_cc, "v_cc"), _num(v_sat, "v_sat"), _num(p, "p")), "calc.rf.pa.load_line_r", ids, "ohm", "R_L = (V_CC - V_sat)^2 / (2 P) (Cripps)")


def quarter_wave_lumped_l(f: Traced[float], z0: Traced[float], ids: tuple[str, str] = ("f", "z0")) -> Traced[float]:
    """The lumped lambda/4's series inductor Z0 / w (H)."""
    return _derived(quarter_wave_lumped_l_h(_num(f, "f"), _num(z0, "z0")), "calc.rf.quarter_wave_lumped.l", ids, "H", "L = Z0 / (2 pi f) (C-L-C pi equivalent of lambda/4)")


def quarter_wave_lumped_c(f: Traced[float], z0: Traced[float], ids: tuple[str, str] = ("f", "z0")) -> Traced[float]:
    """The lumped lambda/4's shunt capacitors 1 / (w Z0) (F)."""
    return _derived(quarter_wave_lumped_c_f(_num(f, "f"), _num(z0, "z0")), "calc.rf.quarter_wave_lumped.c", ids, "F", "C = 1 / (2 pi f Z0) (C-L-C pi equivalent of lambda/4)")


def quad_phase(f: Traced[float], r_source: Traced[float], c_q: Traced[float], l: Traced[float], q_u: Traced[float], f_q: Traced[float], c_p: Traced[float],
               c_trim: Traced[float], r_p: Traced[float], ids: tuple[str, ...] = ("f", "r_source", "c_q", "l", "q_u", "f_q", "c_p", "c_trim", "r_p")) -> Traced[float]:
    """The quadrature network's exact phase at f (deg)."""
    value = quad_phase_deg(_num(f, "f"), _num(r_source, "r_source"), _num(c_q, "c_q"), _num(l, "l"), _num(q_u, "q_u"), _num(f_q, "f_q"), _num(c_p, "c_p"),
                           _num(c_trim, "c_trim"), _num(r_p, "r_p"))
    return _derived(value, "calc.rf.quad.phase", ids, "deg", "phi = arg V(quad) / V_s of R_s - C_q into L (2 pi f_q L / Q_u) // (C_p + C_trim) // R_p (exact)")


def bjt_bias_ic(v_b: Traced[float], v_be: Traced[float], r_e: Traced[float], ids: tuple[str, str, str] = ("v_b", "v_be", "r_e")) -> Traced[float]:
    """A divider-biased BJT's collector current (V_B - V_BE) / R_E (A)."""
    return _derived(bjt_bias_ic_a(_num(v_b, "v_b"), _num(v_be, "v_be"), _num(r_e, "r_e")), "calc.rf.bjt_bias.ic", ids, "A", "I_C ~ (V_B - V_BE) / R_E (beta -> infinity; unloaded divider)")


def limiter_level(v_d: Traced[float], gain: Traced[float], ids: tuple[str, str] = ("v_d", "gain")) -> Traced[float]:
    """A diode limiter's clip level V_D x gain (V)."""
    return _derived(limiter_level_v(_num(v_d, "v_d"), _num(gain, "gain")), "calc.rf.limiter.level", ids, "V", "V_lim = V_D x gain")


def db_sum(a: Traced[float], b: Traced[float], ids: tuple[str, str] = ("a", "b")) -> Traced[float]:
    """The sum of two levels / gains in dB."""
    return _derived(db_sum_value(_num(a, "a"), _num(b, "b")), "calc.rf.db_sum", ids, "dB", "x_dB = a_dB + b_dB")


def emphasis_corner(tau: Traced[float], ids: tuple[str] = ("tau",)) -> Traced[float]:
    """The emphasis corner 1 / (2 pi tau) (Hz)."""
    return _derived(emphasis_corner_hz(_num(tau, "tau")), "calc.audio.emphasis.corner", ids, "Hz", "f_c = 1 / (2 pi tau)")


def emphasis_db_at(f: Traced[float], f_c: Traced[float], sign: Traced[float], ids: tuple[str, str, str] = ("f", "f_c", "sign")) -> Traced[float]:
    """The emphasis term sign x 10 log10(1 + (f/f_c)^2) (dB)."""
    return _derived(emphasis_db(_num(f, "f"), _num(f_c, "f_c"), _num(sign, "sign")), "calc.audio.emphasis.db_at", ids, "dB", "A = sign 10 log10(1 + (f / f_c)^2) (+1 pre-, -1 de-emphasis)")


def highpass1_db_at(f: Traced[float], f_c: Traced[float], ids: tuple[str, str] = ("f", "f_c")) -> Traced[float]:
    """A first-order high-pass response (dB)."""
    return _derived(highpass1_db(_num(f, "f"), _num(f_c, "f_c")), "calc.audio.highpass1.db_at", ids, "dB", "A = -10 log10(1 + (f_c / f)^2)")


def lowpass1_db_at(f: Traced[float], f_c: Traced[float], ids: tuple[str, str] = ("f", "f_c")) -> Traced[float]:
    """A first-order low-pass response (dB)."""
    return _derived(lowpass1_db(_num(f, "f"), _num(f_c, "f_c")), "calc.audio.lowpass1.db_at", ids, "dB", "A = -10 log10(1 + (f / f_c)^2)")


def lossy_integrator_db_at(f: Traced[float], tau_i: Traced[float], r_in: Traced[float], r_dc: Traced[float],
                           ids: tuple[str, str, str, str] = ("f", "tau_i", "r_in", "r_dc")) -> Traced[float]:
    """The lossy integrator's magnitude at f (dB)."""
    value = lossy_integrator_db(_num(f, "f"), _num(tau_i, "tau_i"), _num(r_in, "r_in"), _num(r_dc, "r_dc"))
    return _derived(value, "calc.audio.lossy_integrator.db_at", ids, "dB", "|H| = (R_dc / R_in) / sqrt(1 + (w tau_i R_dc / R_in)^2), tau_i = R_in C")


def butterworth_q(n: Traced[float], k: Traced[float], ids: tuple[str, str] = ("n", "k")) -> Traced[float]:
    """The Butterworth k-th pole pair's Q = 1/g_k."""
    return _derived(butterworth_pair_q(_num(n, "n"), _num(k, "k")), "calc.audio.butterworth.q", ids, None, "Q_k = 1 / g_k = 1 / (2 sin((2k - 1) pi / (2n)))")


def sallen_key_c1(q: Traced[float], f_c: Traced[float], r: Traced[float], ids: tuple[str, str, str] = ("q", "f_c", "r")) -> Traced[float]:
    """The unity-gain Sallen-Key low-pass feedback capacitor (F)."""
    return _derived(sallen_key_c1_f(_num(q, "q"), _num(f_c, "f_c"), _num(r, "r")), "calc.audio.sallen_key.c1", ids, "F", "C1 = 2 Q / (2 pi f_c R) (unity gain, R1 = R2 = R)")


def sallen_key_c2(q: Traced[float], f_c: Traced[float], r: Traced[float], ids: tuple[str, str, str] = ("q", "f_c", "r")) -> Traced[float]:
    """The unity-gain Sallen-Key low-pass capacitor to ground (F)."""
    return _derived(sallen_key_c2_f(_num(q, "q"), _num(f_c, "f_c"), _num(r, "r")), "calc.audio.sallen_key.c2", ids, "F", "C2 = 1 / (2 Q 2 pi f_c R) (unity gain, R1 = R2 = R)")


def tot_period(k_rc: Traced[float], r: Traced[float], c: Traced[float], ids: tuple[str, str, str] = ("k_rc", "r", "c")) -> Traced[float]:
    """The 4060 time-out 2^13 k_RC R C (s)."""
    return _derived(tot_period_s(_num(k_rc, "k_rc"), _num(r, "r"), _num(c, "c")), "calc.rf.tot.period", ids, "s", "T = 2^13 k_RC R C (Q14; k_RC a stated choice)")


def power_rail_budget(i_load: Traced[float], i_rating: Traced[float], ids: tuple[str, str] = ("i_load", "i_rating")) -> Traced[float]:
    """The regulator's rating margin I_rating - I_load (A)."""
    return _derived(rail_budget_a(_num(i_load, "i_load"), _num(i_rating, "i_rating")), "calc.power.rail_budget", ids, "A", "margin = I_rating - I_load")


def regulator_headroom(v_in_min: Traced[float], v_out: Traced[float], v_dropout: Traced[float], i_load: Traced[float], r_path: Traced[float],
                       ids: tuple[str, ...] = ("v_in_min", "v_out", "v_dropout", "i_load", "r_path")) -> Traced[float]:
    """The regulator's headroom at the minimum input (V)."""
    value = regulator_headroom_v(_num(v_in_min, "v_in_min"), _num(v_out, "v_out"), _num(v_dropout, "v_dropout"), _num(i_load, "i_load"), _num(r_path, "r_path"))
    return _derived(value, "calc.regulator.headroom", ids, "V", "headroom = V_in,min - I_load R_path - V_dropout - V_out")


#: tool id -> the traced calculator (registered in :data:`ai_eda.tools.calc.recompute.CALCULATORS` with the roles of
#: :data:`ai_eda.tools.calc.basic.ROLES`)
RADIO_CALCULATORS = {
    "calc.rf.mult.stage": mult_stage,
    "calc.rf.mult.spur": mult_spur,
    "calc.rf.harmonic": harmonic,
    "calc.rf.superhet.lo": superhet_lo,
    "calc.rf.superhet.image": superhet_image,
    "calc.rf.superhet.half_if": superhet_half_if,
    "calc.rf.superhet.second_image": superhet_second_image,
    "calc.rf.superhet.lo_spur_response": superhet_lo_spur_response,
    "calc.rf.fm.obw99": fm_obw99,
    "calc.rf.fm.pm_integrator_tau": fm_pm_integrator_tau,
    "calc.rf.fm.pm_drive_limit": fm_pm_drive_limit,
    "calc.rf.varactor.c_at_bias": varactor_c_at_bias,
    "calc.rf.varactor.dc_dv": varactor_dc_dv,
    "calc.rf.pm.k_pm": pm_k_pm,
    "calc.rf.pm.c_fixed": pm_c_fixed,
    "calc.rf.pm.source_r": pm_source_r,
    "calc.rf.pm.source_r_loaded": pm_source_r_loaded,
    "calc.rf.pm.tank_phase": pm_tank_phase,
    "calc.rf.pm.tank_phase_loaded": pm_tank_phase_loaded,
    "calc.rf.q_parallel": q_parallel,
    "calc.rf.resonator.top_c.bw_for_qe": top_c_bw_for_qe,
    "calc.rf.resonator.top_c.c_couple": top_c_c_couple,
    "calc.rf.resonator.top_c.c_tap": top_c_c_tap,
    "calc.rf.resonator.top_c.c_shunt": top_c_c_shunt,
    "calc.rf.resonator.top_c.s21_db": top_c_s21_db,
    "calc.rf.resonator.top_c.rel_s21_db": top_c_rel_s21_db,
    "calc.rf.resonator.top_c.port_r": top_c_port_r,
    "calc.rf.resonator.top_c.port_x": top_c_port_x,
    "calc.rf.resonator.top_c.c_tap_reactive": top_c_c_tap_reactive,
    "calc.rf.resonator.top_c.ported_s21_db": top_c_ported_s21_db,
    "calc.rf.resonator.top_c.ported_rel_s21_db": top_c_ported_rel_s21_db,
    "calc.rf.bpf.dissipation_loss": bpf_dissipation_loss,
    "calc.rf.resonator.single_tuned.rejection": single_tuned_rejection,
    "calc.rf.resonator.single_tuned.insertion_loss": single_tuned_insertion_loss,
    "calc.rf.resonator.double_tuned.rejection": double_tuned_rejection,
    "calc.crystal.ladder.k": ladder_k,
    "calc.crystal.ladder.q": ladder_q,
    "calc.crystal.ladder.center": ladder_center,
    "calc.crystal.ladder.c_couple": ladder_c_couple,
    "calc.crystal.ladder.r_end": ladder_r_end,
    "calc.crystal.ladder.mesh_c": ladder_mesh_c,
    "calc.crystal.ladder.s21_db": ladder_s21_db,
    "calc.rf.attenuator.pi.r_shunt": attenuator_pi_r_shunt,
    "calc.rf.attenuator.pi.r_series": attenuator_pi_r_series,
    "calc.rf.pa.load_line_r": pa_load_line_r,
    "calc.rf.quarter_wave_lumped.l": quarter_wave_lumped_l,
    "calc.rf.quarter_wave_lumped.c": quarter_wave_lumped_c,
    "calc.rf.quad.phase": quad_phase,
    "calc.rf.bjt_bias.ic": bjt_bias_ic,
    "calc.rf.limiter.level": limiter_level,
    "calc.rf.db_sum": db_sum,
    "calc.audio.emphasis.corner": emphasis_corner,
    "calc.audio.emphasis.db_at": emphasis_db_at,
    "calc.audio.highpass1.db_at": highpass1_db_at,
    "calc.audio.lowpass1.db_at": lowpass1_db_at,
    "calc.audio.lossy_integrator.db_at": lossy_integrator_db_at,
    "calc.audio.butterworth.q": butterworth_q,
    "calc.audio.sallen_key.c1": sallen_key_c1,
    "calc.audio.sallen_key.c2": sallen_key_c2,
    "calc.rf.tot.period": tot_period,
    "calc.power.rail_budget": power_rail_budget,
    "calc.regulator.headroom": regulator_headroom,
}


__all__ = [
    "DB_PER_NEPER",
    "HALF_POWER_DB",
    "LADDER_MAX_ORDER",
    "LADDER_MIN_ORDER",
    "MAX_FRACTIONAL_BW",
    "MAX_PLAN_ORDER",
    "OBW_MAX_INDEX",
    "OBW_POWER_FRACTION",
    "RADIO_CALCULATORS",
    "RESONATOR_MAX_ORDER",
    "TOT_Q14_PERIODS",
    "CrystalLadder",
    "PortedTopC",
    "TopCNetwork",
    "attenuator_pi_r_series",
    "attenuator_pi_r_shunt",
    "bessel_j_values",
    "bjt_bias_ic",
    "bjt_bias_ic_a",
    "bpf_dissipation_loss",
    "butterworth_pair_q",
    "butterworth_q",
    "crystal_ladder",
    "crystal_ladder_center_hz",
    "crystal_ladder_s21",
    "crystal_ladder_s21_db_value",
    "db_sum",
    "db_sum_value",
    "dissipation_loss_db",
    "double_tuned_rejection",
    "double_tuned_rejection_db",
    "emphasis_corner",
    "emphasis_corner_hz",
    "emphasis_db",
    "emphasis_db_at",
    "fm_obw99",
    "fm_obw99_hz",
    "fm_pm_drive_limit",
    "fm_pm_integrator_tau",
    "harmonic",
    "harmonic_hz",
    "highpass1_db",
    "highpass1_db_at",
    "ladder_c_couple",
    "ladder_center",
    "ladder_k",
    "ladder_k_values",
    "ladder_mesh_c",
    "ladder_q",
    "ladder_r_end",
    "ladder_s21_db",
    "limiter_level",
    "limiter_level_v",
    "lossy_integrator_db",
    "lossy_integrator_db_at",
    "lowpass1_db",
    "lowpass1_db_at",
    "mult_spur",
    "mult_stage",
    "multiplier_spur_hz",
    "multiplier_stage_hz",
    "pa_load_line_ohm",
    "pa_load_line_r",
    "pi_pad_r_series_ohm",
    "pi_pad_r_shunt_ohm",
    "pm_c_fixed",
    "pm_c_fixed_f",
    "pm_drive_limit_v",
    "pm_integrator_tau_s",
    "pm_k_pm",
    "pm_k_pm_rad_per_v",
    "pm_source_r",
    "pm_source_r_loaded",
    "pm_source_r_loaded_ohm",
    "pm_source_r_ohm",
    "pm_tank_phase",
    "pm_tank_phase_deg",
    "pm_tank_phase_loaded",
    "pm_tank_phase_loaded_deg",
    "port_shunt_l_impedance",
    "power_rail_budget",
    "q_parallel",
    "q_parallel_value",
    "quad_phase",
    "quad_phase_deg",
    "quarter_wave_lumped_c",
    "quarter_wave_lumped_c_f",
    "quarter_wave_lumped_l",
    "quarter_wave_lumped_l_h",
    "rail_budget_a",
    "regulator_headroom",
    "regulator_headroom_v",
    "sallen_key_c1",
    "sallen_key_c1_f",
    "sallen_key_c2",
    "sallen_key_c2_f",
    "single_tuned_insertion_loss",
    "single_tuned_insertion_loss_db",
    "single_tuned_rejection",
    "single_tuned_rejection_db",
    "superhet_half_if",
    "superhet_half_if_hz",
    "superhet_image",
    "superhet_image_hz",
    "superhet_lo",
    "superhet_lo_hz",
    "superhet_lo_spur_response",
    "superhet_lo_spur_response_hz",
    "superhet_second_image",
    "superhet_second_image_hz",
    "top_c_bw_for_qe",
    "top_c_bw_for_qe_hz",
    "top_c_c_couple",
    "top_c_c_shunt",
    "top_c_c_tap",
    "top_c_c_tap_reactive",
    "top_c_network",
    "top_c_port_r",
    "top_c_port_r_ohm",
    "top_c_port_x",
    "top_c_port_x_ohm",
    "top_c_ported_network",
    "top_c_ported_rel_s21_db",
    "top_c_ported_rel_s21_db_value",
    "top_c_ported_s21_complex",
    "top_c_ported_s21_db",
    "top_c_ported_s21_db_value",
    "top_c_rel_s21_db",
    "top_c_rel_s21_db_value",
    "top_c_s21_complex",
    "top_c_s21_db",
    "top_c_s21_db_value",
    "top_c_tap_reactive_f",
    "tot_period",
    "tot_period_s",
    "varactor_c_at_bias",
    "varactor_c_f",
    "varactor_dc_dv",
    "varactor_slope_f_per_v",
]
