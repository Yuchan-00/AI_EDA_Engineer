"""RF calculators: levels in dB, reflection and matching, low-pass prototypes, wavelengths, link budget, noise, modulation.

Invariant: every RF number a design rests on comes from one of the closed
forms below, each registered like every other calculator (``calc.rf.*`` and
``calc.crystal.c_for_load`` in :data:`~ai_eda.tools.calc.basic.ROLES` /
:data:`~ai_eda.tools.calc.basic.ROLE_UNITS`, :data:`~ai_eda.tools.calc.recompute.CALCULATORS`,
``CALC_VERSION`` 0.9), so ``calc.recompute`` re-derives a stored value from the
ids its provenance names. A calculator refuses (``ValueError`` with a sentence)
every input outside the formula's domain and every result that is not a finite
number (``'... overflows'`` / ``'... underflows'``) - never an extrapolated or
infinite value. Beside each ``Traced`` calculator stands a plain-float helper
that computes the same number (the traced one calls it).

Units (the strings the calculators write and :data:`~ai_eda.tools.calc.basic.ROLE_UNITS`
expects): levels ``dBm`` (dB re 1 mW), ``dBW`` (dB re 1 W), ``dB`` (a ratio),
``dBc`` (re the carrier), ``dBi`` (antenna gain re isotropic), ``dBuV/m``
(field strength re 1 uV/m); ``W``, ``V``, ``Hz``, ``ohm``, ``H``, ``F``, ``m``
(free-space lengths), ``mm`` (board lengths, as :mod:`~ai_eda.tools.calc.tline`),
``K``, ``V/m``, ``ppm``, ``percent``; ``None`` for a plain ratio, a reflection
coefficient magnitude, a Q, a prototype g-value, a filter order, a modulation
index or a velocity factor. A level is never converted silently: dBm -> W is
``calc.rf.dbm_to_w`` and nothing else.

Constants: c0 = 299 792 458 m/s (exact, SI; :data:`~ai_eda.tools.calc.tline.C0`),
k_B = 1.380649e-23 J/K (exact, SI 2019). :data:`T0_K` = 290 K is the reference
noise temperature of the noise figure (IEEE); the noise calculators take the
temperature as an input - 290 K is a choice a template states, never a default.

Formulas and references:

* Levels (IEC 60027-3; the dBm / dBW / dB definitions): P_W = 1e-3 10^(P_dBm/10),
  P_dBm = 10 log10(P / 1 mW); P_W = 10^(P_dBW/10); a power ratio 10^(x/10), a
  voltage (field) ratio 10^(x/20); the peak voltage of a sine of power P into R,
  V_pk = sqrt(2 R P).
* Reflection (D. M. Pozar, "Microwave Engineering", 4th ed., Wiley 2012, 2.3):
  |Gamma| = |(Z_L - Z0)/(Z_L + Z0)| for a load Z_L = R + jX on a real Z0 > 0;
  VSWR = (1 + |Gamma|)/(1 - |Gamma|); return loss RL = -20 log10 |Gamma|;
  mismatch loss ML = -10 log10(1 - |Gamma|^2).
* L-match (C. Bowick, "RF Circuit Design", 2nd ed., Newnes 2008, ch. 4; Pozar
  5.1), real source and load resistances only: Q = sqrt(R_hi/R_lo - 1); the
  shunt element stands across the higher resistance, the series element on the
  lower-resistance side. Low-pass form: L_series = Q R_lo / w, C_shunt =
  Q / (w R_hi); high-pass form: C_series = 1/(w Q R_lo), L_shunt = R_hi/(w Q);
  w = 2 pi f.
* Pi-match (Bowick ch. 4, the virtual-resistance method): R_v = R_hi/(Q^2 + 1)
  with the loaded Q given at the higher-resistance end; Q_2 = sqrt(R_lo/R_v - 1)
  at the other end; each end's shunt capacitor has X_C = R_end/Q_end; the series
  inductor L = (Q + Q_2) R_v / w. Q must exceed sqrt(R_hi/R_lo - 1) (else R_v is
  not below R_lo and no second section exists).
* Low-pass prototypes (Pozar 8.3; G. L. Matthaei, L. Young, E. M. T. Jones,
  "Microwave Filters, Impedance-Matching Networks, and Coupling Structures",
  1964, 4.05). Butterworth (maximally flat): g_k = 2 sin((2k - 1) pi/(2n)),
  g_{n+1} = 1; f_c is the -3 dB frequency; attenuation 10 log10(1 + (f/f_c)^(2n)).
  Chebyshev (equal ripple L_r dB): beta = ln coth(L_r/(40/ln 10)) (the tables'
  "17.37" is 40/ln 10 = 17.3718, used exactly - with 17.37 the 3 dB n = 3 g1
  reads 3.3489 instead of the tabulated 3.3487), gamma = sinh(beta/(2n)),
  a_k = sin((2k - 1) pi/(2n)), b_k = gamma^2 + sin^2(k pi/n), g_1 = 2 a_1/gamma,
  g_k = 4 a_{k-1} a_k/(b_{k-1} g_{k-1}); g_{n+1} = 1 for n odd and coth^2(beta/4)
  (!= 1) for n even. g_{n+1} is a load resistance when g_n is a shunt C and a
  load conductance when g_n is a series L (Pozar 8.3): on the shunt-C-first
  ladder scaled here an even order ends in a series L, so its load is
  Z0/g_{n+1} (a conductance g_{n+1}/Z0); on the dual, series-L-first ladder it
  is g_{n+1} Z0. Either way it is not Z0, so a filter between two equal Z0
  terminations uses n odd (``tests/test_rf_calc.py`` checks n = 2, 4, 6 with
  the load Z0/g_{n+1} against the attenuation below). Here f_c is
  the edge of the equal-ripple band (not the -3 dB point); attenuation
  10 log10(1 + eps^2 T_n(f/f_c)^2), eps^2 = 10^(L_r/10) - 1, T_n(x) =
  cos(n acos x) for x <= 1 and cosh(n acosh x) above. Scaling to a ladder
  starting with a shunt capacitor (Pozar 8.4): C = g/(2 pi f_c Z0),
  L = g Z0/(2 pi f_c). Orders 1..:data:`LPF_MAX_ORDER`.
* Wavelengths: lambda = c0/f; a quarter-wave monopole l = vf c0/(4 f) with a
  stated velocity factor vf (never defaulted); the guided wavelength on a board
  lambda_g = c0/(f sqrt(e_eff)) with e_eff from ``calc.tline.microstrip.e_eff``
  (quasi-static, as that calculator).
* Link budget (H. T. Friis, "A Note on a Simple Transmission Formula",
  Proc. IRE 34, 1946; ITU-R P.525): free-space path loss 20 log10(4 pi d f/c0)
  (far field only; a distance inside lambda/(4 pi), where the formula would
  give a gain, is refused); EIRP = P_tx + G_tx - L_tx (dBm, dBi, dB);
  P_rx = EIRP - L_path + G_rx - L_rx; margin = P_rx - sensitivity.
* ERP (ITU Radio Regulations Nos. 1.161-1.163; ERC/REC 70-03 and ETSI
  EN 300 220 / EN 300 296 write e.r.p. = e.i.r.p. - 2.15 dB): the effective
  radiated power is referenced to a half-wave dipole, whose gain over the
  isotropic radiator is :data:`DIPOLE_GAIN_DBI` = 2.15 dBi (the thin half-wave
  dipole's directivity 4/Cin(2 pi) = 1.6409 is 2.1509 dB; regulators use 2.15,
  and so does this module). EIRP = ERP + 2.15, ERP = EIRP - 2.15 (dBm), and
  ERP = P_tx + G_tx - L_tx - 2.15 for a conducted power and an antenna gain in
  dBi - the only bridge between an ``erp`` limit and a computed number (a
  gain written in dBd is not read by the quantity parser).
* Noise (Nyquist / Johnson; Pozar 10.1): kTB in dBm = 10 log10(k_B T B/1 mW)
  (-173.98 dBm at 290 K in 1 Hz); noise floor = kTB + NF; sensitivity =
  kTB + NF + SNR_required; the Friis cascade (H. T. Friis, "Noise Figures of
  Radio Receivers", Proc. IRE 32, 1944) F = F_1 + (F_rest - 1)/G_1 in linear
  terms, applied from the back of the chain one stage at a time.
* Modulation (e.g. S. Haykin, "Communication Systems"): full-carrier DSB AM
  occupies 2 f_m, carries P_c (1 + m^2/2), each sideband at 20 log10(m/2) dBc;
  the index m is the depth in percent / 100 (``calc.rf.am.index_from_depth``:
  the ``modulation_depth`` requirement is read in percent, the AM calculators
  take the ratio). FM: Carson's rule B = 2 (delta_f + f_m) (J. R. Carson, 1922),
  beta = delta_f/f_m.
* Field strength (far field, free space): E = sqrt(30 EIRP)/d (V/m, W, m) -
  30 is the conventional eta0/(4 pi) with eta0 taken as 120 pi (29.98 with
  eta0 = 376.73 ohm; the 0.07 % difference is the convention field-strength
  limits are written in); EIRP = (E d)^2/30; E_V/m = 1e-6 10^(E_dBuV/m / 20).
* Resonance and Q: L = 1/((2 pi f)^2 C), C = 1/((2 pi f)^2 L) (the resonance
  frequency itself is ``calc.lc.cutoff``); a series L with resistance R has
  Q = 2 pi f L/R; the -3 dB bandwidth of a resonator f0/Q; a ppm offset
  f ppm 1e-6. Crystal load: ``calc.crystal.c_for_load`` C1 = C2 =
  2 (C_L - C_stray), the inverse of ``calc.crystal.load_capacitance``.

Not modelled anywhere here: conductor, dielectric and radiation losses, the
parasitics of real inductors and capacitors (self-resonance, ESR, pad
capacitance), complex source / load impedances for the matching networks (a
reactive antenna must first be resonated or absorbed by a human's design),
antenna efficiency and pattern, multipath and fading beyond free space, and
the finite Q of a filter's elements. A number from this module is a design
target; only a measurement (or a SPICE run of the network with the parts'
own models) says what the built circuit does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ai_eda.ir.provenance import Traced
from ai_eda.tools.calc.basic import _derived
from ai_eda.tools.calc.tline import C0

#: Boltzmann constant, J/K (exact by the SI 2019 definition)
K_B = 1.380649e-23
#: the reference noise temperature of the noise figure (IEEE), K - a choice a caller states, never a default
T0_K = 290.0
#: the half-wave dipole's gain over isotropic, dBi: the reference of ERP (ITU RR 1.161-1.163; ETSI / CEPT write
#: e.r.p. = e.i.r.p. - 2.15 dB; the thin dipole's directivity 4/Cin(2 pi) = 1.6409 is 2.1509 dB, regulators round to 2.15)
DIPOLE_GAIN_DBI = 2.15
#: the far-field factor of E = sqrt(30 P G)/d: eta0/(4 pi) with eta0 taken as 120 pi (the convention of field-strength limits)
FIELD_FACTOR = 30.0
#: the "17.37" of the Chebyshev tables, exactly: 40 / ln 10
CHEBYSHEV_RIPPLE_CONSTANT = 40.0 / math.log(10.0)
#: highest low-pass prototype order the g-value and attenuation calculators accept (Pozar's tables list n = 1..10;
#: the cap keeps the Chebyshev recurrence bounded and stays in the orders a board filter uses)
LPF_MAX_ORDER = 20
#: one milliwatt and one microvolt per metre, the references of dBm and dBuV/m
MILLIWATT = 1e-3
MICROVOLT_PER_M = 1e-6
#: 10^x overflows a double above this exponent
_MAX_EXP10 = 308.0


# --------------------------------------------------------------------------- input and range checks


def _num(t: Traced, what: str) -> float:
    """The traced number (never a bool or text, always finite)."""
    v = t.value
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{what} must be a number, got {v!r}")
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"{what} must be a finite number, got {v!r}")
    return x


def _positive(x: float, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0.0:
        raise ValueError(f"{what} must be positive, got {x!r}")
    return float(x)


def _non_negative(x: float, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0.0:
        raise ValueError(f"{what} must not be negative, got {x!r}")
    return float(x)


def _finite(x: float, what: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError(f"{what} must be a finite number, got {x!r}")
    return float(x)


def _integer(x: float, what: str, lo: int, hi: int) -> int:
    """An integer-valued number in ``lo..hi`` (a filter order, an element index); refused otherwise."""
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not float(x).is_integer():
        raise ValueError(f"{what} must be an integer, got {x!r}")
    n = int(x)
    if not lo <= n <= hi:
        raise ValueError(f"{what} must lie in {lo}..{hi}, got {n}")
    return n


def _pow10(x: float, what: str) -> float:
    """10^x, refusing an exponent whose power is not a finite double."""
    if x > _MAX_EXP10:
        raise ValueError(f"{what} overflows: 10^{x:.6g} is not a finite number")
    return 10.0 ** x


def _log10_positive(x: float, what: str) -> float:
    if not math.isfinite(x) or x <= 0.0:
        raise ValueError(f"{what} must be positive to take its logarithm, got {x!r}")
    return math.log10(x)


def _nonzero_result(x: float, what: str) -> float:
    """A result that underflowed to 0 (or is not finite) is refused: it would read as 'no power' / 'no value'."""
    if not math.isfinite(x):
        raise ValueError(f"{what} overflows: the result is not a finite number for these inputs")
    if x == 0.0:
        raise ValueError(f"{what} underflows: the result rounds to 0 in float arithmetic for these inputs")
    return x


def _db_of_one_plus(ln_y: float) -> float:
    """10 log10(1 + y) for y = exp(ln_y), without overflow for a huge y (softplus in natural log)."""
    if ln_y > 0.0:
        ln_one_plus = ln_y + math.log1p(math.exp(-ln_y))
    else:
        ln_one_plus = math.log1p(math.exp(ln_y))
    return 10.0 / math.log(10.0) * ln_one_plus


# --------------------------------------------------------------------------- levels (plain helpers)


def dbm_to_watts(p_dbm: float) -> float:
    """P_W = 1e-3 10^(P_dBm/10)."""
    return _nonzero_result(MILLIWATT * _pow10(_finite(p_dbm, "the level in dBm") / 10.0, "calc.rf.dbm_to_w"), "calc.rf.dbm_to_w")


def watts_to_dbm(p_w: float) -> float:
    """P_dBm = 10 log10(P / 1 mW) for P > 0."""
    return 10.0 * _log10_positive(_positive(p_w, "the power") / MILLIWATT, "the power")


def dbw_to_watts(p_dbw: float) -> float:
    """P_W = 10^(P_dBW/10)."""
    return _nonzero_result(_pow10(_finite(p_dbw, "the level in dBW") / 10.0, "calc.rf.dbw_to_w"), "calc.rf.dbw_to_w")


def watts_to_dbw(p_w: float) -> float:
    """P_dBW = 10 log10(P / 1 W) for P > 0."""
    return 10.0 * _log10_positive(_positive(p_w, "the power"), "the power")


def db_to_power(x_db: float) -> float:
    """A power ratio 10^(x/10)."""
    return _nonzero_result(_pow10(_finite(x_db, "the ratio in dB") / 10.0, "calc.rf.db_to_power_ratio"), "calc.rf.db_to_power_ratio")


def power_to_db(ratio: float) -> float:
    """10 log10(ratio) of a power ratio > 0."""
    return 10.0 * _log10_positive(_positive(ratio, "a power ratio"), "a power ratio")


def db_to_voltage(x_db: float) -> float:
    """A voltage (field) ratio 10^(x/20)."""
    return _nonzero_result(_pow10(_finite(x_db, "the ratio in dB") / 20.0, "calc.rf.db_to_voltage_ratio"), "calc.rf.db_to_voltage_ratio")


def voltage_to_db(ratio: float) -> float:
    """20 log10(ratio) of a voltage ratio > 0."""
    return 20.0 * _log10_positive(_positive(ratio, "a voltage ratio"), "a voltage ratio")


def dbm_to_peak_volts(p_dbm: float, r_ohm: float) -> float:
    """The peak voltage of a sine of power P (dBm) into R: V_pk = sqrt(2 R P)."""
    r = _positive(r_ohm, "the resistance")
    return math.sqrt(2.0 * r * dbm_to_watts(p_dbm))


# --------------------------------------------------------------------------- reflection (plain helpers)


def reflection_magnitude(r_load: float, x_load: float, z0: float) -> float:
    """|Gamma| = |(Z_L - Z0)/(Z_L + Z0)| for Z_L = R + jX on a real Z0 > 0 (R >= 0: a passive load)."""
    z_ref = _positive(z0, "Z0")
    r = _non_negative(r_load, "the load resistance")
    zl = complex(r, _finite(x_load, "the load reactance"))
    return abs((zl - z_ref) / (zl + z_ref))


def _gamma_below_one(g: float, what: str) -> float:
    if not math.isfinite(g) or g < 0.0:
        raise ValueError(f"|Gamma| must lie in 0..1, got {g!r}")
    if g >= 1.0:
        raise ValueError(f"|Gamma| = {g:.6g} is total reflection: the {what} is infinite")
    return g


def vswr_from_gamma(g: float) -> float:
    """VSWR = (1 + |Gamma|)/(1 - |Gamma|), 0 <= |Gamma| < 1."""
    g = _gamma_below_one(g, "VSWR")
    return (1.0 + g) / (1.0 - g)


def return_loss_db(g: float) -> float:
    """RL = -20 log10 |Gamma|, 0 < |Gamma| <= 1."""
    if not math.isfinite(g) or g < 0.0 or g > 1.0:
        raise ValueError(f"|Gamma| must lie in 0..1, got {g!r}")
    if g == 0.0:
        raise ValueError("|Gamma| = 0 is a perfect match: the return loss is infinite")
    return -20.0 * math.log10(g)


def mismatch_loss_db(g: float) -> float:
    """ML = -10 log10(1 - |Gamma|^2), 0 <= |Gamma| < 1."""
    g = _gamma_below_one(g, "mismatch loss")
    return -10.0 * math.log10(1.0 - g * g)


# --------------------------------------------------------------------------- matching networks (plain helpers)


def _omega(f_hz: float) -> float:
    return 2.0 * math.pi * _positive(f_hz, "the frequency")


def _hi_lo(r_source: float, r_load: float) -> tuple[float, float]:
    rs = _positive(r_source, "the source resistance")
    rl = _positive(r_load, "the load resistance")
    return max(rs, rl), min(rs, rl)


def lmatch_q_value(r_source: float, r_load: float) -> float:
    """The loaded Q of an L-match between two real resistances: Q = sqrt(R_hi/R_lo - 1)."""
    r_hi, r_lo = _hi_lo(r_source, r_load)
    if r_hi == r_lo:
        raise ValueError("equal resistances need no L-match")
    return math.sqrt(r_hi / r_lo - 1.0)


@dataclass(frozen=True)
class LMatch:
    """An L-match between two real resistances.

    ``series`` is the series element (on the lower-resistance side), ``shunt``
    the shunt element across the higher resistance: in the low-pass form a
    series inductor (H) and a shunt capacitor (F), in the high-pass form a
    series capacitor (F) and a shunt inductor (H). ``shunt_at`` names the side
    the shunt element stands across (``"source"`` or ``"load"``).
    """

    q: float
    series: float
    shunt: float
    shunt_at: str
    form: str


def lmatch(f_hz: float, r_source: float, r_load: float, form: str) -> LMatch:
    """The L-match of ``form`` ``"lowpass"`` (series L, shunt C) or ``"highpass"`` (series C, shunt L) at ``f_hz``."""
    q = lmatch_q_value(r_source, r_load)
    w = _omega(f_hz)
    r_hi, r_lo = _hi_lo(r_source, r_load)
    shunt_at = "source" if r_source > r_load else "load"
    if form == "lowpass":
        series, shunt = q * r_lo / w, q / (w * r_hi)
    elif form == "highpass":
        series, shunt = 1.0 / (w * q * r_lo), r_hi / (w * q)
    else:
        raise ValueError(f"form must be 'lowpass' or 'highpass', got {form!r}")
    return LMatch(q=q, series=_nonzero_result(series, f"calc.rf.lmatch.{form} series element"), shunt=_nonzero_result(shunt, f"calc.rf.lmatch.{form} shunt element"), shunt_at=shunt_at, form=form)


@dataclass(frozen=True)
class PiMatch:
    """A pi network (shunt C, series L, shunt C) between two real resistances.

    ``q_source`` / ``q_load`` are the loaded Qs at each end (the given Q at
    the higher-resistance end, Q_2 at the other), ``r_virtual`` the virtual
    resistance at the inductor's middle, ``c_source`` / ``c_load`` the shunt
    capacitors (F) and ``l_series`` the inductor (H).
    """

    q_source: float
    q_load: float
    r_virtual: float
    c_source: float
    l_series: float
    c_load: float


def pi_match(f_hz: float, r_source: float, r_load: float, q: float) -> PiMatch:
    """The pi-match of loaded Q ``q`` (at the higher-resistance end) between ``r_source`` and ``r_load`` at ``f_hz``."""
    w = _omega(f_hz)
    r_hi, r_lo = _hi_lo(r_source, r_load)
    q_hi = _positive(q, "Q")
    q_min = math.sqrt(r_hi / r_lo - 1.0)
    if q_hi <= q_min:
        raise ValueError(f"Q must exceed sqrt(R_hi/R_lo - 1) = {q_min:.6g} for a pi-match between {r_hi:.6g} and {r_lo:.6g} ohm (else the virtual resistance is not below {r_lo:.6g} ohm)")
    r_v = r_hi / (q_hi * q_hi + 1.0)
    q_lo = math.sqrt(r_lo / r_v - 1.0)
    c_hi = q_hi / (w * r_hi)  # X_C = R_hi / Q  ->  C = Q / (w R_hi)
    c_lo = q_lo / (w * r_lo)
    l = (q_hi + q_lo) * r_v / w
    source_is_hi = r_source >= r_load
    return PiMatch(
        q_source=q_hi if source_is_hi else q_lo, q_load=q_lo if source_is_hi else q_hi, r_virtual=r_v,
        c_source=_nonzero_result(c_hi if source_is_hi else c_lo, "calc.rf.pimatch.c_source"),
        l_series=_nonzero_result(l, "calc.rf.pimatch.l_series"),
        c_load=_nonzero_result(c_lo if source_is_hi else c_hi, "calc.rf.pimatch.c_load"),
    )


# --------------------------------------------------------------------------- low-pass prototypes (plain helpers)


def butterworth_g_values(n: int) -> list[float]:
    """[g_1 .. g_{n+1}] of the Butterworth (maximally flat) low-pass prototype of order ``n``."""
    order = _integer(n, "the filter order n", 1, LPF_MAX_ORDER)
    return [2.0 * math.sin((2 * k - 1) * math.pi / (2 * order)) for k in range(1, order + 1)] + [1.0]


def _chebyshev_beta(ripple_db: float) -> float:
    r = _positive(ripple_db, "the passband ripple (dB)")
    x = r / CHEBYSHEV_RIPPLE_CONSTANT
    beta = math.log(1.0 / math.tanh(x))
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"a passband ripple of {r:.6g} dB is outside what the Chebyshev prototype formula resolves in float arithmetic (coth(ripple/17.37) rounds to 1)")
    return beta


def chebyshev_g_values(n: int, ripple_db: float) -> list[float]:
    """[g_1 .. g_{n+1}] of the Chebyshev (equal-ripple ``ripple_db``) low-pass prototype of order ``n``."""
    order = _integer(n, "the filter order n", 1, LPF_MAX_ORDER)
    beta = _chebyshev_beta(ripple_db)
    gamma = math.sinh(beta / (2 * order))
    a = [math.sin((2 * k - 1) * math.pi / (2 * order)) for k in range(1, order + 1)]
    b = [gamma * gamma + math.sin(k * math.pi / order) ** 2 for k in range(1, order + 1)]
    g = [2.0 * a[0] / gamma]
    for k in range(2, order + 1):
        g.append(4.0 * a[k - 2] * a[k - 1] / (b[k - 2] * g[k - 2]))
    g.append(1.0 if order % 2 == 1 else 1.0 / math.tanh(beta / 4.0) ** 2)
    for i, value in enumerate(g, start=1):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"the Chebyshev prototype g_{i} is not a positive finite number for n = {order}, ripple {ripple_db:.6g} dB")
    return g


def lpf_shunt_c_farads(g: float, f_c: float, z0: float) -> float:
    """A prototype's shunt element scaled to Z0 and f_c: C = g/(2 pi f_c Z0)."""
    gv = _positive(g, "the prototype value g")
    return _nonzero_result(gv / (_omega(f_c) * _positive(z0, "Z0")), "calc.rf.lpf.shunt_c")


def lpf_series_l_henries(g: float, f_c: float, z0: float) -> float:
    """A prototype's series element scaled to Z0 and f_c: L = g Z0/(2 pi f_c)."""
    gv = _positive(g, "the prototype value g")
    return _nonzero_result(gv * _positive(z0, "Z0") / _omega(f_c), "calc.rf.lpf.series_l")


def butterworth_attenuation_db(n: int, f: float, f_c: float) -> float:
    """The Butterworth prototype's insertion loss at ``f`` (f_c the -3 dB frequency): 10 log10(1 + (f/f_c)^(2n))."""
    order = _integer(n, "the filter order n", 1, LPF_MAX_ORDER)
    x = _non_negative(f, "the frequency") / _positive(f_c, "the cutoff frequency")
    if x == 0.0:
        return 0.0
    return _db_of_one_plus(2 * order * math.log(x))


def chebyshev_attenuation_db(n: int, ripple_db: float, f: float, f_c: float) -> float:
    """The Chebyshev prototype's insertion loss at ``f`` (f_c the ripple-band edge): 10 log10(1 + eps^2 T_n(f/f_c)^2)."""
    order = _integer(n, "the filter order n", 1, LPF_MAX_ORDER)
    r = _positive(ripple_db, "the passband ripple (dB)")
    x = _non_negative(f, "the frequency") / _positive(f_c, "the cutoff frequency")
    eps2 = _pow10(r / 10.0, "the ripple factor") - 1.0
    if eps2 <= 0.0:
        raise ValueError(f"a passband ripple of {r:.6g} dB rounds to no ripple in float arithmetic")
    if x <= 1.0:
        t = math.cos(order * math.acos(x))
        return 10.0 * math.log10(1.0 + eps2 * t * t)
    u = order * math.acosh(x)
    ln_t = u + math.log1p(math.exp(-2.0 * u)) - math.log(2.0)  # ln cosh(u), no overflow
    return _db_of_one_plus(math.log(eps2) + 2.0 * ln_t)


# --------------------------------------------------------------------------- wavelengths, link budget, noise (plain helpers)


def wavelength_m(f_hz: float) -> float:
    """lambda = c0 / f (m)."""
    return _nonzero_result(C0 / _positive(f_hz, "the frequency"), "calc.rf.wavelength")


def quarter_wave_m(f_hz: float, vf: float) -> float:
    """A quarter-wave monopole's length l = vf c0/(4 f) (m); ``vf`` a stated velocity factor in (0, 1]."""
    v = _finite(vf, "the velocity factor")
    if not 0.0 < v <= 1.0:
        raise ValueError(f"the velocity factor must lie in (0, 1], got {vf!r}")
    return _nonzero_result(v * C0 / (4.0 * _positive(f_hz, "the frequency")), "calc.rf.quarter_wave")


def lambda_g_mm(f_hz: float, e_eff: float) -> float:
    """The guided wavelength on a line of effective permittivity e_eff: 1000 c0/(f sqrt(e_eff)) (mm)."""
    e = _finite(e_eff, "the effective permittivity")
    if e < 1.0:
        raise ValueError(f"an effective permittivity is at least 1, got {e_eff!r}")
    return _nonzero_result(1000.0 * C0 / (_positive(f_hz, "the frequency") * math.sqrt(e)), "calc.rf.lambda_g")


def fspl_db(f_hz: float, d_m: float) -> float:
    """Free-space path loss 20 log10(4 pi d f / c0) (dB); a distance inside lambda/(4 pi) is refused (far field only)."""
    arg = 4.0 * math.pi * _positive(d_m, "the distance") * _positive(f_hz, "the frequency") / C0
    if not math.isfinite(arg):
        raise ValueError("calc.rf.fspl overflows: 4 pi d f / c0 is not a finite number for these inputs")
    if arg <= 1.0:
        raise ValueError(f"the distance {d_m:.6g} m is inside lambda/(4 pi) at {f_hz:.6g} Hz: the free-space path-loss formula holds only in the far field")
    return 20.0 * math.log10(arg)


def eirp_dbm(p_tx_dbm: float, g_tx_dbi: float, l_tx_db: float) -> float:
    """EIRP = P_tx + G_tx - L_tx (dBm)."""
    return _finite(p_tx_dbm, "the transmit power") + _finite(g_tx_dbi, "the antenna gain") - _non_negative(l_tx_db, "the feed loss")


def eirp_dbm_from_erp(erp_dbm: float) -> float:
    """EIRP = ERP + 2.15 dB (dBm): the half-wave dipole's gain over isotropic, :data:`DIPOLE_GAIN_DBI`."""
    return _finite(erp_dbm, "the ERP") + DIPOLE_GAIN_DBI


def erp_dbm_from_eirp(eirp_dbm_value: float) -> float:
    """ERP = EIRP - 2.15 dB (dBm)."""
    return _finite(eirp_dbm_value, "the EIRP") - DIPOLE_GAIN_DBI


def erp_dbm(p_tx_dbm: float, g_tx_dbi: float, l_tx_db: float) -> float:
    """ERP = P_tx + G_tx - L_tx - 2.15 (dBm; the antenna gain in dBi)."""
    return eirp_dbm(p_tx_dbm, g_tx_dbi, l_tx_db) - DIPOLE_GAIN_DBI


def received_power_dbm(eirp: float, path_loss_db: float, g_rx_dbi: float, l_rx_db: float) -> float:
    """P_rx = EIRP - L_path + G_rx - L_rx (dBm)."""
    return _finite(eirp, "the EIRP") - _non_negative(path_loss_db, "the path loss") + _finite(g_rx_dbi, "the antenna gain") - _non_negative(l_rx_db, "the feed loss")


def link_margin_db(p_rx_dbm: float, sensitivity_dbm: float) -> float:
    """margin = P_rx - sensitivity (dB)."""
    return _finite(p_rx_dbm, "the received power") - _finite(sensitivity_dbm, "the sensitivity")


def thermal_noise_dbm(t_k: float, b_hz: float) -> float:
    """kTB in dBm: 10 log10(k_B T B / 1 mW)."""
    p = K_B * _positive(t_k, "the noise temperature") * _positive(b_hz, "the bandwidth")
    return 10.0 * _log10_positive(p / MILLIWATT, "kTB")


def noise_floor_dbm(t_k: float, b_hz: float, nf_db: float) -> float:
    """kTB + NF (dBm)."""
    return thermal_noise_dbm(t_k, b_hz) + _non_negative(nf_db, "the noise figure")


def friis_nf_db(nf1_db: float, g1_db: float, nf_rest_db: float) -> float:
    """Two-stage Friis cascade in dB: F = F1 + (F_rest - 1)/G1 (linear). Cascade a longer chain from the back."""
    f1 = _pow10(_non_negative(nf1_db, "the first stage's noise figure") / 10.0, "the noise factor")
    fr = _pow10(_non_negative(nf_rest_db, "the rest's noise figure") / 10.0, "the noise factor")
    g1 = _nonzero_result(_pow10(_finite(g1_db, "the first stage's gain") / 10.0, "the gain"), "the first stage's linear gain")
    return 10.0 * _log10_positive(f1 + (fr - 1.0) / g1, "the cascade noise factor")


def sensitivity_dbm(t_k: float, b_hz: float, nf_db: float, snr_db: float) -> float:
    """kTB + NF + SNR_required (dBm)."""
    return noise_floor_dbm(t_k, b_hz, nf_db) + _finite(snr_db, "the required SNR")


# --------------------------------------------------------------------------- modulation, field strength, resonance (plain helpers)


def _am_index(m: float, *, zero_ok: bool) -> float:
    x = _finite(m, "the modulation index m")
    if x < 0.0 or x > 1.0 or (x == 0.0 and not zero_ok):
        raise ValueError(f"the AM modulation index must lie in {'[0' if zero_ok else '(0'}, 1] (above 1 the envelope is over-modulated and the formula does not hold), got {m!r}")
    return x


def am_bandwidth_hz(f_m: float) -> float:
    """Full-carrier DSB AM occupies 2 f_m (the highest modulating frequency f_m; ideal, no splatter)."""
    return 2.0 * _positive(f_m, "the modulating frequency")


def am_total_power_w(p_c: float, m: float) -> float:
    """P_total = P_c (1 + m^2/2) for 0 <= m <= 1."""
    return _positive(p_c, "the carrier power") * (1.0 + _am_index(m, zero_ok=True) ** 2 / 2.0)


def am_sideband_dbc_value(m: float) -> float:
    """Each sideband's level 20 log10(m/2) dBc for 0 < m <= 1 (no sideband at m = 0)."""
    x = _finite(m, "the modulation index m")
    if x == 0.0:
        raise ValueError("at m = 0 there is no sideband: its level is minus infinity")
    return 20.0 * math.log10(_am_index(x, zero_ok=False) / 2.0)


def am_index_from_depth_value(depth_percent: float) -> float:
    """m = depth / 100 for a depth in (0, 100] percent."""
    d = _finite(depth_percent, "the modulation depth")
    if not 0.0 < d <= 100.0:
        raise ValueError(f"the AM modulation depth must lie in (0, 100] percent, got {depth_percent!r}")
    return d / 100.0


def carson_bandwidth_hz(delta_f: float, f_m: float) -> float:
    """Carson's rule B = 2 (delta_f + f_m)."""
    return 2.0 * (_non_negative(delta_f, "the peak deviation") + _positive(f_m, "the modulating frequency"))


def fm_index_value(delta_f: float, f_m: float) -> float:
    """beta = delta_f / f_m."""
    return _non_negative(delta_f, "the peak deviation") / _positive(f_m, "the modulating frequency")


def field_strength_vm(eirp_w: float, d_m: float) -> float:
    """Far-field free-space E = sqrt(30 EIRP)/d (V/m)."""
    return _nonzero_result(math.sqrt(FIELD_FACTOR * _positive(eirp_w, "the EIRP")) / _positive(d_m, "the distance"), "calc.rf.field_strength")


def eirp_w_from_field(e_vm: float, d_m: float) -> float:
    """EIRP = (E d)^2 / 30 (W), the inverse of :func:`field_strength_vm`."""
    ed = _positive(e_vm, "the field strength") * _positive(d_m, "the distance")
    return _nonzero_result(ed * ed / FIELD_FACTOR, "calc.rf.eirp_from_field")


def dbuvm_to_volts_per_m(e_dbuvm: float) -> float:
    """E = 1e-6 10^(E_dBuV/m / 20) (V/m)."""
    return _nonzero_result(MICROVOLT_PER_M * _pow10(_finite(e_dbuvm, "the level in dBuV/m") / 20.0, "calc.rf.dbuvm_to_vm"), "calc.rf.dbuvm_to_vm")


def volts_per_m_to_dbuvm(e_vm: float) -> float:
    """E_dBuV/m = 20 log10(E / 1 uV/m) for E > 0."""
    return 20.0 * _log10_positive(_positive(e_vm, "the field strength") / MICROVOLT_PER_M, "the field strength")


def _resonance_denominator(f_hz: float, x: float, what: str, tool: str) -> float:
    w = _omega(f_hz)
    d = w * w * _positive(x, what)
    if d == 0.0:
        raise ValueError(f"{tool} underflows: (2 pi f)^2 {what} is zero in float arithmetic for these inputs")
    return d


def l_for_resonance_h(f_hz: float, c_f: float) -> float:
    """L = 1/((2 pi f)^2 C)."""
    return _nonzero_result(1.0 / _resonance_denominator(f_hz, c_f, "the capacitance", "calc.rf.lc.l_for_resonance"), "calc.rf.lc.l_for_resonance")


def c_for_resonance_f(f_hz: float, l_h: float) -> float:
    """C = 1/((2 pi f)^2 L)."""
    return _nonzero_result(1.0 / _resonance_denominator(f_hz, l_h, "the inductance", "calc.rf.lc.c_for_resonance"), "calc.rf.lc.c_for_resonance")


def q_series_value(f_hz: float, l_h: float, r_ohm: float) -> float:
    """Q of an inductor L with series resistance R at f: 2 pi f L / R."""
    return _nonzero_result(_omega(f_hz) * _positive(l_h, "the inductance") / _positive(r_ohm, "the series resistance"), "calc.rf.q_series")


def bandwidth_from_q_hz(f0_hz: float, q: float) -> float:
    """The -3 dB bandwidth of a resonator: f0 / Q."""
    return _nonzero_result(_positive(f0_hz, "the centre frequency") / _positive(q, "Q"), "calc.rf.bandwidth_from_q")


def ppm_offset_hz(f_hz: float, ppm: float) -> float:
    """The frequency offset f ppm 1e-6 (signed as the ppm figure)."""
    return _positive(f_hz, "the frequency") * _finite(ppm, "the ppm figure") * 1e-6


def crystal_c_for_load_f(c_load: float, c_stray: float) -> float:
    """C1 = C2 = 2 (C_L - C_stray): the equal load capacitors that present C_L to a crystal (inverse of calc.crystal.load_capacitance)."""
    cl = _positive(c_load, "the crystal's load capacitance")
    cs = _non_negative(c_stray, "the stray capacitance")
    if cl <= cs:
        raise ValueError(f"the stray capacitance {cs:.6g} F already reaches the load capacitance {cl:.6g} F: no capacitor pair can add the rest")
    return 2.0 * (cl - cs)


# --------------------------------------------------------------------------- traced calculators (registered)


def dbm_to_w(p_dbm: Traced[float], ids: tuple[str] = ("p_dbm",)) -> Traced[float]:
    """dBm -> W: P = 1e-3 10^(P_dBm/10)."""
    return _derived(dbm_to_watts(_num(p_dbm, "p_dbm")), "calc.rf.dbm_to_w", ids, "W", "P = 1e-3 * 10^(P_dBm / 10)")


def w_to_dbm(p: Traced[float], ids: tuple[str] = ("p",)) -> Traced[float]:
    """W -> dBm: 10 log10(P / 1 mW)."""
    return _derived(watts_to_dbm(_num(p, "p")), "calc.rf.w_to_dbm", ids, "dBm", "P_dBm = 10 log10(P / 1 mW)")


def dbw_to_w(p_dbw: Traced[float], ids: tuple[str] = ("p_dbw",)) -> Traced[float]:
    """dBW -> W: P = 10^(P_dBW/10)."""
    return _derived(dbw_to_watts(_num(p_dbw, "p_dbw")), "calc.rf.dbw_to_w", ids, "W", "P = 10^(P_dBW / 10)")


def w_to_dbw(p: Traced[float], ids: tuple[str] = ("p",)) -> Traced[float]:
    """W -> dBW: 10 log10(P / 1 W)."""
    return _derived(watts_to_dbw(_num(p, "p")), "calc.rf.w_to_dbw", ids, "dBW", "P_dBW = 10 log10(P / 1 W)")


def db_to_power_ratio(x_db: Traced[float], ids: tuple[str] = ("x_db",)) -> Traced[float]:
    """A power ratio from dB: 10^(x/10)."""
    return _derived(db_to_power(_num(x_db, "x_db")), "calc.rf.db_to_power_ratio", ids, None, "ratio = 10^(x_dB / 10)")


def power_ratio_to_db(ratio: Traced[float], ids: tuple[str] = ("ratio",)) -> Traced[float]:
    """dB of a power ratio: 10 log10(r)."""
    return _derived(power_to_db(_num(ratio, "ratio")), "calc.rf.power_ratio_to_db", ids, "dB", "x_dB = 10 log10(ratio)")


def db_to_voltage_ratio(x_db: Traced[float], ids: tuple[str] = ("x_db",)) -> Traced[float]:
    """A voltage ratio from dB: 10^(x/20)."""
    return _derived(db_to_voltage(_num(x_db, "x_db")), "calc.rf.db_to_voltage_ratio", ids, None, "ratio = 10^(x_dB / 20)")


def voltage_ratio_to_db(ratio: Traced[float], ids: tuple[str] = ("ratio",)) -> Traced[float]:
    """dB of a voltage ratio: 20 log10(r)."""
    return _derived(voltage_to_db(_num(ratio, "ratio")), "calc.rf.voltage_ratio_to_db", ids, "dB", "x_dB = 20 log10(ratio)")


def dbm_to_vpeak(p_dbm: Traced[float], r: Traced[float], ids: tuple[str, str] = ("p_dbm", "r")) -> Traced[float]:
    """The peak voltage of a sine of power P (dBm) into R: sqrt(2 R P)."""
    return _derived(dbm_to_peak_volts(_num(p_dbm, "p_dbm"), _num(r, "r")), "calc.rf.dbm_to_vpeak", ids, "V", "V_pk = sqrt(2 R P), P = 1e-3 * 10^(P_dBm / 10)")


def gamma_mag(r_load: Traced[float], x_load: Traced[float], z0: Traced[float], ids: tuple[str, str, str] = ("r_load", "x_load", "z0")) -> Traced[float]:
    """|Gamma| of a load R + jX on a real Z0 (Pozar 2.3)."""
    value = reflection_magnitude(_num(r_load, "r_load"), _num(x_load, "x_load"), _num(z0, "z0"))
    return _derived(value, "calc.rf.gamma_mag", ids, None, "|Gamma| = |(Z_L - Z0) / (Z_L + Z0)|, Z_L = R + jX")


def vswr(gamma: Traced[float], ids: tuple[str] = ("gamma",)) -> Traced[float]:
    """VSWR = (1 + |Gamma|)/(1 - |Gamma|)."""
    return _derived(vswr_from_gamma(_num(gamma, "gamma")), "calc.rf.vswr", ids, None, "VSWR = (1 + |Gamma|) / (1 - |Gamma|)")


def return_loss(gamma: Traced[float], ids: tuple[str] = ("gamma",)) -> Traced[float]:
    """Return loss -20 log10 |Gamma| (dB)."""
    return _derived(return_loss_db(_num(gamma, "gamma")), "calc.rf.return_loss", ids, "dB", "RL = -20 log10 |Gamma|")


def mismatch_loss(gamma: Traced[float], ids: tuple[str] = ("gamma",)) -> Traced[float]:
    """Mismatch loss -10 log10(1 - |Gamma|^2) (dB)."""
    return _derived(mismatch_loss_db(_num(gamma, "gamma")), "calc.rf.mismatch_loss", ids, "dB", "ML = -10 log10(1 - |Gamma|^2)")


def lmatch_q(r_source: Traced[float], r_load: Traced[float], ids: tuple[str, str] = ("r_source", "r_load")) -> Traced[float]:
    """The L-match's loaded Q: sqrt(R_hi/R_lo - 1)."""
    return _derived(lmatch_q_value(_num(r_source, "r_source"), _num(r_load, "r_load")), "calc.rf.lmatch.q", ids, None, "Q = sqrt(R_hi / R_lo - 1)")


def _lm(f: Traced, r_source: Traced, r_load: Traced, form: str) -> LMatch:
    return lmatch(_num(f, "f"), _num(r_source, "r_source"), _num(r_load, "r_load"), form)


def lmatch_lowpass_l_series(f: Traced[float], r_source: Traced[float], r_load: Traced[float], ids: tuple[str, str, str] = ("f", "r_source", "r_load")) -> Traced[float]:
    """Low-pass L-match: the series inductor on the lower-resistance side, L = Q R_lo / w (H)."""
    return _derived(_lm(f, r_source, r_load, "lowpass").series, "calc.rf.lmatch.lowpass.l_series", ids, "H", "L = Q R_lo / (2 pi f), Q = sqrt(R_hi / R_lo - 1) (series, lower-R side)")


def lmatch_lowpass_c_shunt(f: Traced[float], r_source: Traced[float], r_load: Traced[float], ids: tuple[str, str, str] = ("f", "r_source", "r_load")) -> Traced[float]:
    """Low-pass L-match: the shunt capacitor across the higher resistance, C = Q / (w R_hi) (F)."""
    return _derived(_lm(f, r_source, r_load, "lowpass").shunt, "calc.rf.lmatch.lowpass.c_shunt", ids, "F", "C = Q / (2 pi f R_hi), Q = sqrt(R_hi / R_lo - 1) (shunt, across the higher R)")


def lmatch_highpass_c_series(f: Traced[float], r_source: Traced[float], r_load: Traced[float], ids: tuple[str, str, str] = ("f", "r_source", "r_load")) -> Traced[float]:
    """High-pass L-match: the series capacitor on the lower-resistance side, C = 1/(w Q R_lo) (F)."""
    return _derived(_lm(f, r_source, r_load, "highpass").series, "calc.rf.lmatch.highpass.c_series", ids, "F", "C = 1 / (2 pi f Q R_lo), Q = sqrt(R_hi / R_lo - 1) (series, lower-R side)")


def lmatch_highpass_l_shunt(f: Traced[float], r_source: Traced[float], r_load: Traced[float], ids: tuple[str, str, str] = ("f", "r_source", "r_load")) -> Traced[float]:
    """High-pass L-match: the shunt inductor across the higher resistance, L = R_hi/(w Q) (H)."""
    return _derived(_lm(f, r_source, r_load, "highpass").shunt, "calc.rf.lmatch.highpass.l_shunt", ids, "H", "L = R_hi / (2 pi f Q), Q = sqrt(R_hi / R_lo - 1) (shunt, across the higher R)")


def _pm(f: Traced, r_source: Traced, r_load: Traced, q: Traced) -> PiMatch:
    return pi_match(_num(f, "f"), _num(r_source, "r_source"), _num(r_load, "r_load"), _num(q, "q"))


_PI_NOTE = "R_v = R_hi / (Q^2 + 1), Q_2 = sqrt(R_lo / R_v - 1) (Bowick ch. 4; Q at the higher-R end)"


def pimatch_c_source(f: Traced[float], r_source: Traced[float], r_load: Traced[float], q: Traced[float], ids: tuple[str, str, str, str] = ("f", "r_source", "r_load", "q")) -> Traced[float]:
    """Pi-match: the shunt capacitor at the source, X_C = R_source / Q_source (F)."""
    return _derived(_pm(f, r_source, r_load, q).c_source, "calc.rf.pimatch.c_source", ids, "F", f"C_source = Q_source / (2 pi f R_source); {_PI_NOTE}")


def pimatch_l_series(f: Traced[float], r_source: Traced[float], r_load: Traced[float], q: Traced[float], ids: tuple[str, str, str, str] = ("f", "r_source", "r_load", "q")) -> Traced[float]:
    """Pi-match: the series inductor L = (Q + Q_2) R_v / w (H)."""
    return _derived(_pm(f, r_source, r_load, q).l_series, "calc.rf.pimatch.l_series", ids, "H", f"L = (Q + Q_2) R_v / (2 pi f); {_PI_NOTE}")


def pimatch_c_load(f: Traced[float], r_source: Traced[float], r_load: Traced[float], q: Traced[float], ids: tuple[str, str, str, str] = ("f", "r_source", "r_load", "q")) -> Traced[float]:
    """Pi-match: the shunt capacitor at the load, X_C = R_load / Q_load (F)."""
    return _derived(_pm(f, r_source, r_load, q).c_load, "calc.rf.pimatch.c_load", ids, "F", f"C_load = Q_load / (2 pi f R_load); {_PI_NOTE}")


def butterworth_g(n: Traced[float], k: Traced[float], ids: tuple[str, str] = ("n", "k")) -> Traced[float]:
    """The Butterworth prototype's g_k (k = 1 .. n+1): 2 sin((2k - 1) pi/(2n)), g_{n+1} = 1."""
    order = _integer(_num(n, "n"), "the filter order n", 1, LPF_MAX_ORDER)
    index = _integer(_num(k, "k"), "the element index k", 1, order + 1)
    return _derived(butterworth_g_values(order)[index - 1], "calc.rf.lpf.butterworth.g", ids, None, "g_k = 2 sin((2k - 1) pi / (2n)), g_(n+1) = 1 (Pozar 8.3)")


def chebyshev_g(n: Traced[float], k: Traced[float], ripple: Traced[float], ids: tuple[str, str, str] = ("n", "k", "ripple")) -> Traced[float]:
    """The Chebyshev prototype's g_k (k = 1 .. n+1) for an equal ripple in dB (Pozar 8.3, Matthaei 4.05)."""
    order = _integer(_num(n, "n"), "the filter order n", 1, LPF_MAX_ORDER)
    index = _integer(_num(k, "k"), "the element index k", 1, order + 1)
    value = chebyshev_g_values(order, _num(ripple, "ripple"))[index - 1]
    return _derived(value, "calc.rf.lpf.chebyshev.g", ids, None, "beta = ln coth(L_r / (40 / ln 10)), gamma = sinh(beta / 2n), g_1 = 2 a_1 / gamma, g_k = 4 a_(k-1) a_k / (b_(k-1) g_(k-1)); g_(n+1) = 1 (n odd) or coth^2(beta / 4) (n even)")


def lpf_shunt_c(g: Traced[float], f_c: Traced[float], z0: Traced[float], ids: tuple[str, str, str] = ("g", "f_c", "z0")) -> Traced[float]:
    """A prototype element as a shunt capacitor: C = g/(2 pi f_c Z0) (F)."""
    return _derived(lpf_shunt_c_farads(_num(g, "g"), _num(f_c, "f_c"), _num(z0, "z0")), "calc.rf.lpf.shunt_c", ids, "F", "C = g / (2 pi f_c Z0)")


def lpf_series_l(g: Traced[float], f_c: Traced[float], z0: Traced[float], ids: tuple[str, str, str] = ("g", "f_c", "z0")) -> Traced[float]:
    """A prototype element as a series inductor: L = g Z0/(2 pi f_c) (H)."""
    return _derived(lpf_series_l_henries(_num(g, "g"), _num(f_c, "f_c"), _num(z0, "z0")), "calc.rf.lpf.series_l", ids, "H", "L = g Z0 / (2 pi f_c)")


def butterworth_attenuation(n: Traced[float], f: Traced[float], f_c: Traced[float], ids: tuple[str, str, str] = ("n", "f", "f_c")) -> Traced[float]:
    """The Butterworth prototype's analytic attenuation at ``f`` (f_c the -3 dB point) in dB."""
    return _derived(butterworth_attenuation_db(_num(n, "n"), _num(f, "f"), _num(f_c, "f_c")), "calc.rf.lpf.butterworth.attenuation", ids, "dB", "A = 10 log10(1 + (f / f_c)^(2n)) (f_c at -3 dB; lossless prototype)")


def chebyshev_attenuation(n: Traced[float], ripple: Traced[float], f: Traced[float], f_c: Traced[float], ids: tuple[str, str, str, str] = ("n", "ripple", "f", "f_c")) -> Traced[float]:
    """The Chebyshev prototype's analytic attenuation at ``f`` (f_c the ripple-band edge) in dB."""
    value = chebyshev_attenuation_db(_num(n, "n"), _num(ripple, "ripple"), _num(f, "f"), _num(f_c, "f_c"))
    return _derived(value, "calc.rf.lpf.chebyshev.attenuation", ids, "dB", "A = 10 log10(1 + eps^2 T_n(f / f_c)^2), eps^2 = 10^(L_r / 10) - 1 (f_c the ripple-band edge; lossless prototype)")


def wavelength(f: Traced[float], ids: tuple[str] = ("f",)) -> Traced[float]:
    """Free-space wavelength c0/f (m)."""
    return _derived(wavelength_m(_num(f, "f")), "calc.rf.wavelength", ids, "m", "lambda = c0 / f")


def quarter_wave(f: Traced[float], vf: Traced[float], ids: tuple[str, str] = ("f", "vf")) -> Traced[float]:
    """A quarter-wave monopole's length vf c0/(4 f) (m)."""
    return _derived(quarter_wave_m(_num(f, "f"), _num(vf, "vf")), "calc.rf.quarter_wave", ids, "m", "l = vf c0 / (4 f) (velocity factor a stated choice)")


def lambda_g(f: Traced[float], e_eff: Traced[float], ids: tuple[str, str] = ("f", "e_eff")) -> Traced[float]:
    """The guided wavelength on a board line: 1000 c0/(f sqrt(e_eff)) (mm)."""
    return _derived(lambda_g_mm(_num(f, "f"), _num(e_eff, "e_eff")), "calc.rf.lambda_g", ids, "mm", "lambda_g = c0 / (f sqrt(e_eff)) (mm; quasi-static e_eff)")


def fspl(f: Traced[float], d: Traced[float], ids: tuple[str, str] = ("f", "d")) -> Traced[float]:
    """Free-space path loss 20 log10(4 pi d f/c0) (dB, far field)."""
    return _derived(fspl_db(_num(f, "f"), _num(d, "d")), "calc.rf.fspl", ids, "dB", "FSPL = 20 log10(4 pi d f / c0) (Friis 1946; far field, free space)")


def eirp(p_tx: Traced[float], g_tx: Traced[float], l_tx: Traced[float], ids: tuple[str, str, str] = ("p_tx", "g_tx", "l_tx")) -> Traced[float]:
    """EIRP = P_tx + G_tx - L_tx (dBm)."""
    return _derived(eirp_dbm(_num(p_tx, "p_tx"), _num(g_tx, "g_tx"), _num(l_tx, "l_tx")), "calc.rf.eirp", ids, "dBm", "EIRP = P_tx + G_tx - L_tx")


def erp_to_eirp(erp: Traced[float], ids: tuple[str] = ("erp",)) -> Traced[float]:
    """EIRP = ERP + 2.15 dB (dBm)."""
    return _derived(eirp_dbm_from_erp(_num(erp, "erp")), "calc.rf.erp_to_eirp", ids, "dBm", "EIRP = ERP + 2.15 dB (half-wave dipole 2.15 dBi; ITU RR 1.161-1.163)")


def eirp_to_erp(eirp: Traced[float], ids: tuple[str] = ("eirp",)) -> Traced[float]:
    """ERP = EIRP - 2.15 dB (dBm)."""
    return _derived(erp_dbm_from_eirp(_num(eirp, "eirp")), "calc.rf.eirp_to_erp", ids, "dBm", "ERP = EIRP - 2.15 dB (half-wave dipole 2.15 dBi; ITU RR 1.161-1.163)")


def erp(p_tx: Traced[float], g_tx: Traced[float], l_tx: Traced[float], ids: tuple[str, str, str] = ("p_tx", "g_tx", "l_tx")) -> Traced[float]:
    """ERP = P_tx + G_tx - L_tx - 2.15 (dBm; G_tx in dBi)."""
    return _derived(erp_dbm(_num(p_tx, "p_tx"), _num(g_tx, "g_tx"), _num(l_tx, "l_tx")), "calc.rf.erp", ids, "dBm", "ERP = P_tx + G_tx - L_tx - 2.15 (G_tx in dBi; half-wave dipole 2.15 dBi)")


def received_power(eirp: Traced[float], path_loss: Traced[float], g_rx: Traced[float], l_rx: Traced[float], ids: tuple[str, str, str, str] = ("eirp", "path_loss", "g_rx", "l_rx")) -> Traced[float]:
    """P_rx = EIRP - L_path + G_rx - L_rx (dBm)."""
    value = received_power_dbm(_num(eirp, "eirp"), _num(path_loss, "path_loss"), _num(g_rx, "g_rx"), _num(l_rx, "l_rx"))
    return _derived(value, "calc.rf.received_power", ids, "dBm", "P_rx = EIRP - L_path + G_rx - L_rx")


def link_margin(p_rx: Traced[float], sensitivity: Traced[float], ids: tuple[str, str] = ("p_rx", "sensitivity")) -> Traced[float]:
    """Link margin P_rx - sensitivity (dB)."""
    return _derived(link_margin_db(_num(p_rx, "p_rx"), _num(sensitivity, "sensitivity")), "calc.rf.link_margin", ids, "dB", "margin = P_rx - S")


def thermal_noise(t: Traced[float], b: Traced[float], ids: tuple[str, str] = ("t", "b")) -> Traced[float]:
    """kTB (dBm)."""
    return _derived(thermal_noise_dbm(_num(t, "t"), _num(b, "b")), "calc.rf.thermal_noise", ids, "dBm", "kTB = 10 log10(k_B T B / 1 mW), k_B = 1.380649e-23 J/K")


def noise_floor(t: Traced[float], b: Traced[float], nf: Traced[float], ids: tuple[str, str, str] = ("t", "b", "nf")) -> Traced[float]:
    """Noise floor kTB + NF (dBm)."""
    return _derived(noise_floor_dbm(_num(t, "t"), _num(b, "b"), _num(nf, "nf")), "calc.rf.noise_floor", ids, "dBm", "N = kTB + NF")


def friis_nf(nf1: Traced[float], g1: Traced[float], nf_rest: Traced[float], ids: tuple[str, str, str] = ("nf1", "g1", "nf_rest")) -> Traced[float]:
    """Friis cascade of a first stage and the rest (dB)."""
    return _derived(friis_nf_db(_num(nf1, "nf1"), _num(g1, "g1"), _num(nf_rest, "nf_rest")), "calc.rf.friis_nf", ids, "dB", "F = F_1 + (F_rest - 1) / G_1 (linear; Friis 1944)")


def sensitivity(t: Traced[float], b: Traced[float], nf: Traced[float], snr: Traced[float], ids: tuple[str, str, str, str] = ("t", "b", "nf", "snr")) -> Traced[float]:
    """Receiver sensitivity kTB + NF + SNR (dBm)."""
    return _derived(sensitivity_dbm(_num(t, "t"), _num(b, "b"), _num(nf, "nf"), _num(snr, "snr")), "calc.rf.sensitivity", ids, "dBm", "S = kTB + NF + SNR_required")


def am_bandwidth(f_m: Traced[float], ids: tuple[str] = ("f_m",)) -> Traced[float]:
    """Occupied bandwidth of full-carrier DSB AM: 2 f_m (Hz)."""
    return _derived(am_bandwidth_hz(_num(f_m, "f_m")), "calc.rf.am.bandwidth", ids, "Hz", "B = 2 f_m (DSB AM, ideal)")


def am_total_power(p_c: Traced[float], m: Traced[float], ids: tuple[str, str] = ("p_c", "m")) -> Traced[float]:
    """Total AM power P_c (1 + m^2/2) (W)."""
    return _derived(am_total_power_w(_num(p_c, "p_c"), _num(m, "m")), "calc.rf.am.total_power", ids, "W", "P = P_c (1 + m^2 / 2)")


def am_sideband_dbc(m: Traced[float], ids: tuple[str] = ("m",)) -> Traced[float]:
    """Each AM sideband's level 20 log10(m/2) (dBc)."""
    return _derived(am_sideband_dbc_value(_num(m, "m")), "calc.rf.am.sideband_dbc", ids, "dBc", "sideband = 20 log10(m / 2) dBc")


def am_index_from_depth(depth: Traced[float], ids: tuple[str] = ("depth",)) -> Traced[float]:
    """The AM index m from a depth in percent: depth / 100."""
    return _derived(am_index_from_depth_value(_num(depth, "depth")), "calc.rf.am.index_from_depth", ids, None, "m = depth / 100 (depth in percent)")


def fm_carson_bandwidth(delta_f: Traced[float], f_m: Traced[float], ids: tuple[str, str] = ("delta_f", "f_m")) -> Traced[float]:
    """Carson's rule 2 (delta_f + f_m) (Hz)."""
    return _derived(carson_bandwidth_hz(_num(delta_f, "delta_f"), _num(f_m, "f_m")), "calc.rf.fm.carson_bandwidth", ids, "Hz", "B = 2 (delta_f + f_m) (Carson 1922)")


def fm_modulation_index(delta_f: Traced[float], f_m: Traced[float], ids: tuple[str, str] = ("delta_f", "f_m")) -> Traced[float]:
    """The FM index beta = delta_f / f_m."""
    return _derived(fm_index_value(_num(delta_f, "delta_f"), _num(f_m, "f_m")), "calc.rf.fm.modulation_index", ids, None, "beta = delta_f / f_m")


def field_strength(eirp: Traced[float], d: Traced[float], ids: tuple[str, str] = ("eirp", "d")) -> Traced[float]:
    """Far-field E = sqrt(30 EIRP)/d (V/m)."""
    return _derived(field_strength_vm(_num(eirp, "eirp"), _num(d, "d")), "calc.rf.field_strength", ids, "V/m", "E = sqrt(30 EIRP) / d (far field, free space; 30 = eta0 / (4 pi) with eta0 = 120 pi)")


def eirp_from_field(e: Traced[float], d: Traced[float], ids: tuple[str, str] = ("e", "d")) -> Traced[float]:
    """EIRP = (E d)^2/30 (W)."""
    return _derived(eirp_w_from_field(_num(e, "e"), _num(d, "d")), "calc.rf.eirp_from_field", ids, "W", "EIRP = (E d)^2 / 30 (far field, free space)")


def dbuvm_to_vm(e_dbuvm: Traced[float], ids: tuple[str] = ("e_dbuvm",)) -> Traced[float]:
    """dBuV/m -> V/m: 1e-6 10^(E/20)."""
    return _derived(dbuvm_to_volts_per_m(_num(e_dbuvm, "e_dbuvm")), "calc.rf.dbuvm_to_vm", ids, "V/m", "E = 1e-6 * 10^(E_dBuV/m / 20)")


def vm_to_dbuvm(e: Traced[float], ids: tuple[str] = ("e",)) -> Traced[float]:
    """V/m -> dBuV/m: 20 log10(E / 1 uV/m)."""
    return _derived(volts_per_m_to_dbuvm(_num(e, "e")), "calc.rf.vm_to_dbuvm", ids, "dBuV/m", "E_dBuV/m = 20 log10(E / 1 uV/m)")


def lc_l_for_resonance(f: Traced[float], c: Traced[float], ids: tuple[str, str] = ("f", "c")) -> Traced[float]:
    """The inductor that resonates with C at f: 1/((2 pi f)^2 C) (H)."""
    return _derived(l_for_resonance_h(_num(f, "f"), _num(c, "c")), "calc.rf.lc.l_for_resonance", ids, "H", "L = 1 / ((2 pi f)^2 C)")


def lc_c_for_resonance(f: Traced[float], l: Traced[float], ids: tuple[str, str] = ("f", "l")) -> Traced[float]:
    """The capacitor that resonates with L at f: 1/((2 pi f)^2 L) (F)."""
    return _derived(c_for_resonance_f(_num(f, "f"), _num(l, "l")), "calc.rf.lc.c_for_resonance", ids, "F", "C = 1 / ((2 pi f)^2 L)")


def q_series(f: Traced[float], l: Traced[float], r: Traced[float], ids: tuple[str, str, str] = ("f", "l", "r")) -> Traced[float]:
    """Q of an inductor with series resistance: 2 pi f L / R."""
    return _derived(q_series_value(_num(f, "f"), _num(l, "l"), _num(r, "r")), "calc.rf.q_series", ids, None, "Q = 2 pi f L / R")


def bandwidth_from_q(f0: Traced[float], q: Traced[float], ids: tuple[str, str] = ("f0", "q")) -> Traced[float]:
    """A resonator's -3 dB bandwidth f0/Q (Hz)."""
    return _derived(bandwidth_from_q_hz(_num(f0, "f0"), _num(q, "q")), "calc.rf.bandwidth_from_q", ids, "Hz", "B = f0 / Q")


def ppm_offset(f: Traced[float], ppm: Traced[float], ids: tuple[str, str] = ("f", "ppm")) -> Traced[float]:
    """A frequency offset f ppm 1e-6 (Hz)."""
    return _derived(ppm_offset_hz(_num(f, "f"), _num(ppm, "ppm")), "calc.rf.ppm_offset", ids, "Hz", "delta_f = f * ppm * 1e-6")


def crystal_c_for_load(c_load: Traced[float], c_stray: Traced[float], ids: tuple[str, str] = ("c_load", "c_stray")) -> Traced[float]:
    """Equal Pierce load capacitors for a crystal's load capacitance: C1 = C2 = 2 (C_L - C_stray) (F)."""
    return _derived(crystal_c_for_load_f(_num(c_load, "c_load"), _num(c_stray, "c_stray")), "calc.crystal.c_for_load", ids, "F", "C1 = C2 = 2 (C_L - C_stray) (inverse of calc.crystal.load_capacitance)")


#: tool id -> the traced calculator (registered in :data:`ai_eda.tools.calc.recompute.CALCULATORS` with the roles of
#: :data:`ai_eda.tools.calc.basic.ROLES`)
RF_CALCULATORS = {
    "calc.rf.dbm_to_w": dbm_to_w,
    "calc.rf.w_to_dbm": w_to_dbm,
    "calc.rf.dbw_to_w": dbw_to_w,
    "calc.rf.w_to_dbw": w_to_dbw,
    "calc.rf.db_to_power_ratio": db_to_power_ratio,
    "calc.rf.power_ratio_to_db": power_ratio_to_db,
    "calc.rf.db_to_voltage_ratio": db_to_voltage_ratio,
    "calc.rf.voltage_ratio_to_db": voltage_ratio_to_db,
    "calc.rf.dbm_to_vpeak": dbm_to_vpeak,
    "calc.rf.gamma_mag": gamma_mag,
    "calc.rf.vswr": vswr,
    "calc.rf.return_loss": return_loss,
    "calc.rf.mismatch_loss": mismatch_loss,
    "calc.rf.lmatch.q": lmatch_q,
    "calc.rf.lmatch.lowpass.l_series": lmatch_lowpass_l_series,
    "calc.rf.lmatch.lowpass.c_shunt": lmatch_lowpass_c_shunt,
    "calc.rf.lmatch.highpass.c_series": lmatch_highpass_c_series,
    "calc.rf.lmatch.highpass.l_shunt": lmatch_highpass_l_shunt,
    "calc.rf.pimatch.c_source": pimatch_c_source,
    "calc.rf.pimatch.l_series": pimatch_l_series,
    "calc.rf.pimatch.c_load": pimatch_c_load,
    "calc.rf.lpf.butterworth.g": butterworth_g,
    "calc.rf.lpf.chebyshev.g": chebyshev_g,
    "calc.rf.lpf.shunt_c": lpf_shunt_c,
    "calc.rf.lpf.series_l": lpf_series_l,
    "calc.rf.lpf.butterworth.attenuation": butterworth_attenuation,
    "calc.rf.lpf.chebyshev.attenuation": chebyshev_attenuation,
    "calc.rf.wavelength": wavelength,
    "calc.rf.quarter_wave": quarter_wave,
    "calc.rf.lambda_g": lambda_g,
    "calc.rf.fspl": fspl,
    "calc.rf.eirp": eirp,
    "calc.rf.erp_to_eirp": erp_to_eirp,
    "calc.rf.eirp_to_erp": eirp_to_erp,
    "calc.rf.erp": erp,
    "calc.rf.received_power": received_power,
    "calc.rf.link_margin": link_margin,
    "calc.rf.thermal_noise": thermal_noise,
    "calc.rf.noise_floor": noise_floor,
    "calc.rf.friis_nf": friis_nf,
    "calc.rf.sensitivity": sensitivity,
    "calc.rf.am.bandwidth": am_bandwidth,
    "calc.rf.am.total_power": am_total_power,
    "calc.rf.am.sideband_dbc": am_sideband_dbc,
    "calc.rf.am.index_from_depth": am_index_from_depth,
    "calc.rf.fm.carson_bandwidth": fm_carson_bandwidth,
    "calc.rf.fm.modulation_index": fm_modulation_index,
    "calc.rf.field_strength": field_strength,
    "calc.rf.eirp_from_field": eirp_from_field,
    "calc.rf.dbuvm_to_vm": dbuvm_to_vm,
    "calc.rf.vm_to_dbuvm": vm_to_dbuvm,
    "calc.rf.lc.l_for_resonance": lc_l_for_resonance,
    "calc.rf.lc.c_for_resonance": lc_c_for_resonance,
    "calc.rf.q_series": q_series,
    "calc.rf.bandwidth_from_q": bandwidth_from_q,
    "calc.rf.ppm_offset": ppm_offset,
    "calc.crystal.c_for_load": crystal_c_for_load,
}


__all__ = [
    "CHEBYSHEV_RIPPLE_CONSTANT",
    "DIPOLE_GAIN_DBI",
    "FIELD_FACTOR",
    "K_B",
    "LPF_MAX_ORDER",
    "MICROVOLT_PER_M",
    "MILLIWATT",
    "RF_CALCULATORS",
    "T0_K",
    "LMatch",
    "PiMatch",
    "am_bandwidth",
    "am_bandwidth_hz",
    "am_index_from_depth",
    "am_index_from_depth_value",
    "am_sideband_dbc",
    "am_sideband_dbc_value",
    "am_total_power",
    "am_total_power_w",
    "bandwidth_from_q",
    "bandwidth_from_q_hz",
    "butterworth_attenuation",
    "butterworth_attenuation_db",
    "butterworth_g",
    "butterworth_g_values",
    "c_for_resonance_f",
    "carson_bandwidth_hz",
    "chebyshev_attenuation",
    "chebyshev_attenuation_db",
    "chebyshev_g",
    "chebyshev_g_values",
    "crystal_c_for_load",
    "crystal_c_for_load_f",
    "db_to_power",
    "db_to_power_ratio",
    "db_to_voltage",
    "db_to_voltage_ratio",
    "dbm_to_peak_volts",
    "dbm_to_vpeak",
    "dbm_to_w",
    "dbm_to_watts",
    "dbuvm_to_vm",
    "dbuvm_to_volts_per_m",
    "dbw_to_w",
    "dbw_to_watts",
    "eirp",
    "eirp_dbm",
    "eirp_dbm_from_erp",
    "eirp_from_field",
    "eirp_to_erp",
    "eirp_w_from_field",
    "erp",
    "erp_dbm",
    "erp_dbm_from_eirp",
    "erp_to_eirp",
    "field_strength",
    "field_strength_vm",
    "fm_carson_bandwidth",
    "fm_index_value",
    "fm_modulation_index",
    "friis_nf",
    "friis_nf_db",
    "fspl",
    "fspl_db",
    "gamma_mag",
    "l_for_resonance_h",
    "lambda_g",
    "lambda_g_mm",
    "lc_c_for_resonance",
    "lc_l_for_resonance",
    "link_margin",
    "link_margin_db",
    "lmatch",
    "lmatch_highpass_c_series",
    "lmatch_highpass_l_shunt",
    "lmatch_lowpass_c_shunt",
    "lmatch_lowpass_l_series",
    "lmatch_q",
    "lmatch_q_value",
    "lpf_series_l",
    "lpf_series_l_henries",
    "lpf_shunt_c",
    "lpf_shunt_c_farads",
    "mismatch_loss",
    "mismatch_loss_db",
    "noise_floor",
    "noise_floor_dbm",
    "pi_match",
    "pimatch_c_load",
    "pimatch_c_source",
    "pimatch_l_series",
    "power_ratio_to_db",
    "power_to_db",
    "ppm_offset",
    "ppm_offset_hz",
    "q_series",
    "q_series_value",
    "quarter_wave",
    "quarter_wave_m",
    "received_power",
    "received_power_dbm",
    "reflection_magnitude",
    "return_loss",
    "return_loss_db",
    "sensitivity",
    "sensitivity_dbm",
    "thermal_noise",
    "thermal_noise_dbm",
    "voltage_ratio_to_db",
    "voltage_to_db",
    "volts_per_m_to_dbuvm",
    "vswr",
    "vswr_from_gamma",
    "w_to_dbm",
    "w_to_dbw",
    "watts_to_dbm",
    "watts_to_dbw",
    "wavelength",
    "wavelength_m",
]
