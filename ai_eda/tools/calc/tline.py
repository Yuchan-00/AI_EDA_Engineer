"""Transmission-line calculators: characteristic impedance, effective permittivity, propagation delay, critical length.

Invariant: every number a signal-integrity decision rests on comes from one
of the closed forms below, evaluated only inside the range its source
states; outside it the calculator raises :class:`TLineRangeError` (a
``ValueError``) naming the range, never an extrapolated number. The
quasi-static (TEM, zero-frequency) values are computed: dispersion, conductor
and dielectric loss, surface roughness and the solder mask over a microstrip
are **not** modelled - a check that quotes these numbers says so. Every
calculator is registered like the others (:data:`~ai_eda.tools.calc.basic.ROLES`,
:data:`~ai_eda.tools.calc.recompute.CALCULATORS`, ``CALC_VERSION`` 0.8), so
``calc.recompute`` re-derives a stored value from the ids its provenance
names - including stackup numbers (``pcb.stackup.dielectrics[0].er``, see
:meth:`ai_eda.ir.Stackup.lookup`).

Units: widths, spacings and dielectric heights in **mm**, copper thickness
in **um** (as :class:`~ai_eda.ir.Stackup` states them), impedances in ohm,
the propagation delay per length ``t_pd`` in s/m, delays and rise times in s,
lengths along a track in mm. Constants: c0 = 299 792 458 m/s (exact),
mu0 = 1.25663706212e-6 H/m (CODATA 2018), eta0 = mu0 c0 = 376.7303 ohm.

Formulas (u = w/h normalised width, g = s/h normalised gap, t/h normalised thickness):

* ``calc.tline.microstrip.z0`` / ``.e_eff`` - E. Hammerstad and O. Jensen,
  "Accurate Models for Microstrip Computer-Aided Design", IEEE MTT-S Int.
  Microwave Symp. Digest, 1980, pp. 407-409 (as given in the Qucs Technical
  Documentation, "Single microstrip line", eqs. 11.4-11.6, 11.15-11.17,
  11.22-11.25)::

      Z01(u)   = eta0/(2 pi) ln(f(u)/u + sqrt(1 + (2/u)^2)),  f(u) = 6 + (2 pi - 6) exp(-(30.666/u)^0.7528)
      e_eff(u) = (er+1)/2 + (er-1)/2 (1 + 10/u)^(-a(u) b(er))
      a(u) = 1 + ln((u^4 + (u/52)^2)/(u^4 + 0.432))/49 + ln(1 + (u/18.1)^3)/18.7
      b(er) = 0.564 ((er - 0.9)/(er + 3))^0.053
      thickness (Wheeler's correction as refined by H&J):
      du1 = (t/h)/pi ln(1 + 4e/((t/h) coth^2 sqrt(6.517 u))),  dur = du1 (1 + sech sqrt(er - 1))/2
      Z0 = Z01(u + dur) / sqrt(e_eff(u + dur)),  e_eff,t = e_eff(u + dur) (Z01(u + du1)/Z01(u + dur))^2

  Stated accuracy (zero thickness): Z01 better than 0.01 % for u <= 1 and
  0.03 % for u <= 1000; e_eff better than 0.2 % for er < 128 and
  0.01 <= u <= 100. The calculator refuses u outside [0.01, 100], er outside
  [1, 128), t >= h and t >= w (the thin-strip correction is approximate and
  has no stated accuracy). Cross-checked in ``tests/test_tline.py`` against
  H. A. Wheeler, "Transmission-Line Properties of a Strip on a Dielectric
  Sheet on a Plane", IEEE Trans. MTT-25(8), 1977 (within 1 % for
  0.1 <= u <= 10, 2.2 <= er <= 10) and the parallel-plate limit of a wide strip.

* ``calc.tline.stripline.z0`` - a strip centred between two planes a
  distance ``b`` apart (homogeneous dielectric). t = 0: S. B. Cohn,
  "Characteristic Impedance of the Shielded-Strip Transmission Line",
  IRE Trans. MTT-2, 1954, exact conformal map
  ``Z0 = eta0/(4 sqrt(er)) K(k)/K(k')``, k = sech(pi w/(2b)), k' = tanh(pi w/(2b)),
  with K(k)/K(k') = AGM(1, k)/AGM(1, k') (arithmetic-geometric mean, exact).
  t > 0: H. A. Wheeler, "Transmission-Line Properties of a Strip Line
  Between Parallel Planes", IEEE Trans. MTT-26(11), 1978::

      x = t/b, m = 6/(3 + 2x/(1 - x)),  dw/t = (1/pi)(1 - ln((x/(2 - x))^2 + (0.0796 x/(w/b + 1.1 x))^m)/2)
      w' = w + dw,  y = (4/pi)(b - t)/w'
      Z0 = eta0/(4 pi sqrt(er)) ln(1 + y (2y + sqrt((2y)^2 + 6.27)))

  Checked range (the calculator refuses outside it): 0.02 <= w/b <= 5,
  t/b <= 0.25. Over that range Wheeler's form at t -> 0 is within 0.5 % of
  Cohn's exact value (``tests/test_tline.py``). Used for inner-layer signals,
  which the router does not route today; :func:`line_geometry` answers an
  inner signal layer with the reason instead of a b it would have to derive.

* ``calc.tline.edge_coupled_microstrip.*`` - M. Kirschning and R. H. Jansen,
  "Accurate Wide-Range Design Equations for the Frequency-Dependent
  Characteristic of Parallel Coupled Microstrip Lines", IEEE Trans. MTT-32(1),
  1984, pp. 83-90 (errata MTT-33(3), 1985), the static (zero-frequency)
  even / odd mode equations as given in the Qucs Technical Documentation,
  "Parallel coupled microstrip lines", eqs. 11.89-11.99 and 11.118-11.129
  (the ``377 ohm`` there is eta0 here). Validity range stated there:
  0.1 <= u <= 10, 0.1 <= g <= 10, 1 <= er <= 18 (refused outside); stated
  accuracy of the static impedances: better than 0.6 %. ``Z_diff = 2 Z_odd``
  (a pair driven differentially). Strip thickness: R. H. Jansen's even / odd
  equivalent widths (Qucs eqs. 11.180-11.182) ``u_e = u + du (1 - exp(-0.69 du/dt)/2)``,
  ``u_o = u_e + dt`` with ``dt = 2 (t/h)/(g er)`` (stated for s >> 2t: the
  calculator refuses s <= 2t), and each mode evaluated as a zero-thickness
  coupled line of its equivalent width - including the single-line Z_L(0) /
  e_eff(0) the equations refer to. Jansen names Hammerstad-Bekkadal's
  single-strip ``du``; this module uses the H&J mixed-media ``dur`` above, a
  documented choice that makes both modes tend to ``calc.tline.microstrip.z0``
  as the gap grows. At t = 0 the values equal KiCad 10's pcb_calculator
  (Qucs lineage) to 1e-6, and they agree with Hammerstad and Jensen's own
  coupled-line model (1980) within 1 % (even) / 2 % (odd) over the validity
  range (``tests/test_tline.py``).

* ``calc.tline.width_for_z0.microstrip`` / ``.stripline`` and
  ``calc.tline.edge_coupled_microstrip.s_for_zdiff`` / ``.w_for_zdiff`` -
  bisection on the forward formula over its valid range (the result is
  within :data:`SOLVE_REL_TOL` of the root in width / gap; deterministic: the
  same inputs give the same float), refusing a target the formula cannot
  reach inside that range.

* ``calc.tline.tpd`` - t_pd = sqrt(e_eff)/c0 (s/m). ``calc.tline.delay`` -
  t_d = l t_pd (l in mm). ``calc.tline.critical_length`` - l_crit =
  fraction t_r / t_pd (mm): a line shorter than l_crit is electrically short
  by the stated rule. ``fraction`` is a confirmed choice: 1/2 is the rule
  "the round trip 2 l t_pd is shorter than the rise time"
  (:data:`ROUND_TRIP_FRACTION`); the stricter 1/6 (:data:`LUMPED_FRACTION`)
  is the lumped-circuit rule of H. Johnson and M. Graham, "High-Speed Digital
  Design", 1993 (dimensions below 1/6 of the rising edge's length). A rule,
  not a simulation: a longer line is handed to the SPICE check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from ai_eda.ir.provenance import Traced
from ai_eda.tools.calc.basic import _derived

#: speed of light in vacuum, m/s (exact by the SI definition)
C0 = 299_792_458.0
#: vacuum permeability, H/m (CODATA 2018)
MU0 = 1.25663706212e-6
#: impedance of free space, ohm
ETA0 = MU0 * C0

#: Hammerstad-Jensen microstrip: stated range of the zero-thickness formulas (u = w/h) and of er (er < 128)
MICROSTRIP_U_RANGE = (0.01, 100.0)
MICROSTRIP_ER_RANGE = (1.0, 128.0)
#: stripline: the range this module checked (w/b; t/b at most the second number)
STRIPLINE_WB_RANGE = (0.02, 5.0)
STRIPLINE_TB_MAX = 0.25
#: Kirschning-Jansen coupled microstrip: stated validity range (u, g = s/h, er)
COUPLED_U_RANGE = (0.1, 10.0)
COUPLED_G_RANGE = (0.1, 10.0)
COUPLED_ER_RANGE = (1.0, 18.0)
#: relative width of the bracket a bisection stops at
SOLVE_REL_TOL = 1e-6
_SOLVE_MAX_STEPS = 200
#: critical-length fractions: the round-trip rule (default choice) and the stricter lumped-circuit rule
ROUND_TRIP_FRACTION = 0.5
LUMPED_FRACTION = 1.0 / 6.0


class TLineRangeError(ValueError):
    """The inputs lie outside the range the formula states (or this module checked); the message names it."""


def _positive(value: float, what: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{what} must be a positive finite number, got {value!r}")
    return v


def _non_negative(value: float, what: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v < 0.0:
        raise ValueError(f"{what} must be a finite number >= 0, got {value!r}")
    return v


#: relative slack of a range bound: a ratio of two decimal lengths lands one ULP off (0.02 / 0.2 = 0.09999999999999999)
_RANGE_SLACK = 1e-12


def _in_range(value: float, lo: float, hi: float, what: str, *, hi_open: bool = False) -> None:
    lo_s, hi_s = lo * (1.0 - _RANGE_SLACK), hi * (1.0 + _RANGE_SLACK)
    ok = lo_s <= value < hi if hi_open else lo_s <= value <= hi_s
    if not ok:
        raise TLineRangeError(f"{what} = {value:.6g} is outside the formula's range [{lo:g}, {hi:g}{')' if hi_open else ']'}")


# --------------------------------------------------------------------------- microstrip (Hammerstad & Jensen 1980)


def _z01(u: float) -> float:
    """Impedance of a zero-thickness strip over a plane in air (homogeneous medium), H&J eq. 11.4."""
    f = 6.0 + (2.0 * math.pi - 6.0) * math.exp(-((30.666 / u) ** 0.7528))
    return ETA0 / (2.0 * math.pi) * math.log(f / u + math.sqrt(1.0 + 4.0 / (u * u)))


def _ab(u: float, er: float) -> float:
    """The exponent a(u) b(er) of the H&J effective-permittivity formula (eqs. 11.16, 11.17)."""
    a = 1.0 + math.log((u ** 4 + (u / 52.0) ** 2) / (u ** 4 + 0.432)) / 49.0 + math.log(1.0 + (u / 18.1) ** 3) / 18.7
    b = 0.564 * ((er - 0.9) / (er + 3.0)) ** 0.053
    return a * b


def _e_eff0(u: float, er: float) -> float:
    """Zero-thickness effective permittivity, H&J eq. 11.15."""
    return (er + 1.0) / 2.0 + (er - 1.0) / 2.0 * (1.0 + 10.0 / u) ** (-_ab(u, er))


def _du_thickness(u: float, t_h: float, er: float) -> tuple[float, float]:
    """``(du1, dur)``: H&J's normalised width increase of a strip of thickness ``t_h`` (eqs. 11.22, 11.23)."""
    if t_h <= 0.0:
        return 0.0, 0.0
    coth = 1.0 / math.tanh(math.sqrt(6.517 * u))
    du1 = t_h / math.pi * math.log(1.0 + 4.0 * math.e / (t_h * coth * coth))
    dur = 0.5 * du1 * (1.0 + 1.0 / math.cosh(math.sqrt(er - 1.0)))
    return du1, dur


@dataclass(frozen=True)
class MicrostripResult:
    """Quasi-static microstrip values: ``z0_ohm``, ``e_eff`` (thickness-corrected), the normalised width ``u`` and the equivalent widths ``u1`` (air) / ``u_r``."""

    z0_ohm: float
    e_eff: float
    u: float
    u1: float
    u_r: float


def microstrip(w_mm: float, h_mm: float, t_um: float, er: float) -> MicrostripResult:
    """Hammerstad & Jensen microstrip (module docstring); :class:`TLineRangeError` outside the stated range."""
    w = _positive(w_mm, "w")
    h = _positive(h_mm, "h")
    t = _non_negative(t_um, "t") / 1000.0
    er = float(er)
    if not math.isfinite(er):
        raise ValueError(f"er must be finite, got {er!r}")
    _in_range(er, *MICROSTRIP_ER_RANGE, "er", hi_open=True)
    u = w / h
    _in_range(u, *MICROSTRIP_U_RANGE, "u = w/h")
    if t >= h:
        raise TLineRangeError(f"t = {t * 1000:.6g} um is not thinner than the substrate h = {h:.6g} mm (outside the thin-strip correction)")
    if t >= w:
        raise TLineRangeError(f"t = {t * 1000:.6g} um is not thinner than the strip w = {w:.6g} mm (outside the thin-strip correction)")
    du1, dur = _du_thickness(u, t / h, er)
    u1, ur = u + du1, u + dur
    e_r = _e_eff0(ur, er)
    z0 = _z01(ur) / math.sqrt(e_r)
    e_eff = e_r * (_z01(u1) / _z01(ur)) ** 2
    return MicrostripResult(z0_ohm=z0, e_eff=e_eff, u=u, u1=u1, u_r=ur)


# --------------------------------------------------------------------------- stripline (Cohn 1954, Wheeler 1978)


def _agm(a: float, b: float) -> float:
    for _ in range(64):
        a, b = 0.5 * (a + b), math.sqrt(a * b)
        if abs(a - b) <= 1e-16 * a:
            break
    return a


def stripline_impedance(w_mm: float, b_mm: float, t_um: float, er: float) -> float:
    """Z0 of a strip centred between planes ``b_mm`` apart: Cohn's exact form at t = 0, Wheeler (1978) for t > 0."""
    w = _positive(w_mm, "w")
    b = _positive(b_mm, "b")
    t = _non_negative(t_um, "t") / 1000.0
    er = float(er)
    if not math.isfinite(er) or er < 1.0:
        raise TLineRangeError(f"er = {er!r} is below 1 (a relative permittivity is at least 1)")
    _in_range(w / b, *STRIPLINE_WB_RANGE, "w/b")
    if t / b > STRIPLINE_TB_MAX:
        raise TLineRangeError(f"t/b = {t / b:.6g} is outside the checked range [0, {STRIPLINE_TB_MAX:g}]")
    if t == 0.0:
        x = math.pi * w / (2.0 * b)
        k, kp = 1.0 / math.cosh(x), math.tanh(x)
        # K(k)/K(k') = (pi / (2 AGM(1, k'))) / (pi / (2 AGM(1, k))) = AGM(1, k) / AGM(1, k')
        return ETA0 / (4.0 * math.sqrt(er)) * _agm(1.0, k) / _agm(1.0, kp)
    x = t / b
    m = 6.0 / (3.0 + 2.0 * x / (1.0 - x))
    dw = t / math.pi * (1.0 - 0.5 * math.log((x / (2.0 - x)) ** 2 + (0.0796 * x / (w / b + 1.1 * x)) ** m))
    y = (4.0 / math.pi) * (b - t) / (w + dw)
    return ETA0 / (4.0 * math.pi * math.sqrt(er)) * math.log(1.0 + y * (2.0 * y + math.sqrt((2.0 * y) ** 2 + 6.27)))


# --------------------------------------------------------------------------- edge-coupled microstrip (Kirschning & Jansen 1984)


@dataclass(frozen=True)
class CoupledResult:
    """Static even / odd mode values of a symmetric edge-coupled microstrip pair; ``z_diff = 2 z_odd``."""

    z_even: float
    z_odd: float
    e_eff_even: float
    e_eff_odd: float
    u_even: float
    u_odd: float

    @property
    def z_diff(self) -> float:
        return 2.0 * self.z_odd


def _kj_even(u: float, g: float, er: float) -> tuple[float, float]:
    """``(Z_even, e_eff_even)`` of a zero-thickness coupled pair of normalised width ``u`` (Qucs eqs. 11.91-11.94, 11.118-11.122)."""
    v = u * (20.0 + g * g) / (10.0 + g * g) + g * math.exp(-g)
    e_even = 0.5 * (er + 1.0) + 0.5 * (er - 1.0) * (1.0 + 10.0 / v) ** (-_ab(v, er))
    z_l, e_l = _z01(u) / math.sqrt(_e_eff0(u, er)), _e_eff0(u, er)
    q4 = _q4(u, g)
    return z_l * math.sqrt(e_l / e_even) / (1.0 - z_l / ETA0 * math.sqrt(e_l) * q4), e_even


def _q2(g: float) -> float:
    return 1.0 + 0.7519 * g + 0.189 * g ** 2.31


def _q4(u: float, g: float) -> float:
    q1 = 0.8695 * u ** 0.194
    q3 = 0.1975 + (16.6 + (8.4 / g) ** 6) ** -0.387 + math.log(g ** 10 / (1.0 + (g / 3.4) ** 10)) / 241.0
    return 2.0 * q1 / (_q2(g) * (math.exp(-g) * u ** q3 + (2.0 - math.exp(-g)) * u ** -q3))


def _kj_odd(u: float, g: float, er: float) -> tuple[float, float]:
    """``(Z_odd, e_eff_odd)`` of a zero-thickness coupled pair of normalised width ``u`` (Qucs eqs. 11.95-11.99, 11.123-11.129)."""
    e_l = _e_eff0(u, er)
    z_l = _z01(u) / math.sqrt(e_l)
    a_o = 0.7287 * (e_l - 0.5 * (er + 1.0)) * (1.0 - math.exp(-0.179 * u))
    b_o = 0.747 * er / (0.15 + er)
    c_o = b_o - (b_o - 0.207) * math.exp(-0.414 * u)
    d_o = 0.593 + 0.694 * math.exp(-0.562 * u)
    e_odd = (0.5 * (er + 1.0) + a_o - e_l) * math.exp(-c_o * g ** d_o) + e_l
    q5 = 1.794 + 1.14 * math.log(1.0 + 0.638 / (g + 0.517 * g ** 2.43))
    q6 = 0.2305 + math.log(g ** 10 / (1.0 + (g / 5.8) ** 10)) / 281.3 + math.log(1.0 + 0.598 * g ** 1.154) / 5.1
    q7 = (10.0 + 190.0 * g * g) / (1.0 + 82.3 * g ** 3)
    q8 = math.exp(-6.5 - 0.95 * math.log(g) - (g / 0.15) ** 5)
    q9 = math.log(q7) * (q8 + 1.0 / 16.5)
    q10 = _q4(u, g) - q5 / _q2(g) * math.exp(math.log(u) * q6 * u ** -q9)
    return z_l * math.sqrt(e_l / e_odd) / (1.0 - z_l / ETA0 * math.sqrt(e_l) * q10), e_odd


def edge_coupled_microstrip(w_mm: float, s_mm: float, h_mm: float, t_um: float, er: float) -> CoupledResult:
    """Kirschning & Jansen static even / odd values with Jansen's thickness widths (module docstring)."""
    w = _positive(w_mm, "w")
    s = _positive(s_mm, "s")
    h = _positive(h_mm, "h")
    t = _non_negative(t_um, "t") / 1000.0
    er = float(er)
    if not math.isfinite(er):
        raise ValueError(f"er must be finite, got {er!r}")
    _in_range(er, *COUPLED_ER_RANGE, "er")
    u, g = w / h, s / h
    _in_range(u, *COUPLED_U_RANGE, "u = w/h")
    _in_range(g, *COUPLED_G_RANGE, "g = s/h")
    if t >= h or t >= w:
        raise TLineRangeError(f"t = {t * 1000:.6g} um is not thinner than the substrate and the strip (h = {h:.6g} mm, w = {w:.6g} mm)")
    u_e = u_o = u
    if t > 0.0:
        if s <= 2.0 * t:
            raise TLineRangeError(f"s = {s:.6g} mm is not larger than 2 t = {2 * t:.6g} mm: Jansen's thickness correction is stated for s >> 2t")
        _, du = _du_thickness(u, t / h, er)
        dt = 2.0 * (t / h) / (g * er)
        u_e = u + du * (1.0 - 0.5 * math.exp(-0.69 * du / dt))
        u_o = u_e + dt
    z_even, e_even = _kj_even(u_e, g, er)
    z_odd, e_odd = _kj_odd(u_o, g, er)
    return CoupledResult(z_even=z_even, z_odd=z_odd, e_eff_even=e_even, e_eff_odd=e_odd, u_even=u_e, u_odd=u_o)


# --------------------------------------------------------------------------- inverse solutions (bisection on the forward formula)


def _bisect(fn: Callable[[float], float], lo: float, hi: float, target: float, what: str, increasing: bool) -> float:
    """The argument in [lo, hi] where the monotonic ``fn`` equals ``target`` (bracket width <= SOLVE_REL_TOL relative)."""
    f_lo, f_hi = fn(lo), fn(hi)
    low, high = (f_lo, f_hi) if increasing else (f_hi, f_lo)
    if not low <= target <= high:
        raise TLineRangeError(f"{what} {target:.6g} ohm is outside what the formula reaches in its valid range ({low:.6g} .. {high:.6g} ohm)")
    for _ in range(_SOLVE_MAX_STEPS):
        mid = 0.5 * (lo + hi)
        above = fn(mid) > target
        if above == increasing:
            hi = mid
        else:
            lo = mid
        if hi - lo <= SOLVE_REL_TOL * hi:
            break
    return 0.5 * (lo + hi)


def _lower_width(u_min: float, h: float, t_mm: float) -> float:
    """The narrowest width the formula accepts: u_min h, or just above t when the strip would otherwise be thinner than thick."""
    lo = u_min * h
    return lo if lo > t_mm else math.nextafter(t_mm, math.inf)


def solve_microstrip_width(z0_ohm: float, h_mm: float, t_um: float, er: float) -> float:
    """The microstrip width (mm) whose Z0 is ``z0_ohm``; :class:`TLineRangeError` for a target outside u in [0.01, 100]."""
    z0 = _positive(z0_ohm, "z0")
    h = _positive(h_mm, "h")
    t_mm = _non_negative(t_um, "t") / 1000.0
    lo, hi = _lower_width(MICROSTRIP_U_RANGE[0], h, t_mm), MICROSTRIP_U_RANGE[1] * h
    if lo >= hi:
        raise TLineRangeError(f"t = {t_um:.6g} um leaves no valid width on h = {h:.6g} mm")
    return _bisect(lambda w: microstrip(w, h, t_um, er).z0_ohm, lo, hi, z0, "target Z0", increasing=False)


def solve_stripline_width(z0_ohm: float, b_mm: float, t_um: float, er: float) -> float:
    """The centred-stripline width (mm) whose Z0 is ``z0_ohm`` (w/b in the checked range)."""
    z0 = _positive(z0_ohm, "z0")
    b = _positive(b_mm, "b")
    return _bisect(lambda w: stripline_impedance(w, b, t_um, er), STRIPLINE_WB_RANGE[0] * b, STRIPLINE_WB_RANGE[1] * b, z0, "target Z0", increasing=False)


def solve_coupled_spacing(z_diff_ohm: float, w_mm: float, h_mm: float, t_um: float, er: float) -> float:
    """The gap (mm) that gives a pair of width ``w_mm`` the differential impedance ``z_diff_ohm`` (Z_diff grows with the gap)."""
    zd = _positive(z_diff_ohm, "z_diff")
    h = _positive(h_mm, "h")
    t_mm = _non_negative(t_um, "t") / 1000.0
    lo = COUPLED_G_RANGE[0] * h
    if t_mm > 0.0 and lo <= 2.0 * t_mm:
        lo = math.nextafter(2.0 * t_mm, math.inf)
    hi = COUPLED_G_RANGE[1] * h
    if lo >= hi:
        raise TLineRangeError(f"t = {t_um:.6g} um leaves no valid gap on h = {h:.6g} mm")
    return _bisect(lambda s: edge_coupled_microstrip(w_mm, s, h, t_um, er).z_diff, lo, hi, zd, "target Z_diff", increasing=True)


def solve_coupled_width(z_diff_ohm: float, s_mm: float, h_mm: float, t_um: float, er: float) -> float:
    """The width (mm) that gives a pair with gap ``s_mm`` the differential impedance ``z_diff_ohm`` (Z_diff falls as the strips widen)."""
    zd = _positive(z_diff_ohm, "z_diff")
    h = _positive(h_mm, "h")
    t_mm = _non_negative(t_um, "t") / 1000.0
    lo, hi = _lower_width(COUPLED_U_RANGE[0], h, t_mm), COUPLED_U_RANGE[1] * h
    if lo >= hi:
        raise TLineRangeError(f"t = {t_um:.6g} um leaves no valid width on h = {h:.6g} mm")
    return _bisect(lambda w: edge_coupled_microstrip(w, s_mm, h, t_um, er).z_diff, lo, hi, zd, "target Z_diff", increasing=False)


# --------------------------------------------------------------------------- delay and the critical length


def propagation_delay(e_eff: float) -> float:
    """t_pd = sqrt(e_eff) / c0 in s/m."""
    e = float(e_eff)
    if not math.isfinite(e) or e < 1.0:
        raise ValueError(f"an effective permittivity is at least 1, got {e_eff!r}")
    return math.sqrt(e) / C0


def line_delay_s(length_mm: float, t_pd_s_per_m: float) -> float:
    """t_d = l t_pd for a length in mm (s)."""
    return _non_negative(length_mm, "length") / 1000.0 * _positive(t_pd_s_per_m, "t_pd")


def critical_length_mm(t_r_s: float, t_pd_s_per_m: float, fraction: float) -> float:
    """l_crit = fraction t_r / t_pd in mm (module docstring: 1/2 round-trip rule, 1/6 lumped rule)."""
    t_r = _positive(t_r_s, "t_r")
    t_pd = _positive(t_pd_s_per_m, "t_pd")
    f = float(fraction)
    if not math.isfinite(f) or not 0.0 < f <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")
    return f * t_r / t_pd * 1000.0


# --------------------------------------------------------------------------- traced calculators (registered)


def _v(t: Traced) -> float:
    v = t.value
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"expected a number, got {v!r}")
    return float(v)


def microstrip_z0(w: Traced[float], h: Traced[float], t: Traced[float], er: Traced[float], ids: tuple[str, str, str, str] = ("w", "h", "t", "er")) -> Traced[float]:
    """Microstrip Z0 (ohm) from width and substrate height (mm), copper thickness (um) and er: Hammerstad & Jensen 1980."""
    r = microstrip(_v(w), _v(h), _v(t), _v(er))
    return _derived(r.z0_ohm, "calc.tline.microstrip.z0", ids, "ohm", "Z0 = Z01(u_r)/sqrt(e_eff(u_r)) (Hammerstad & Jensen 1980, thickness-corrected width u_r; quasi-static, no solder mask)")


def microstrip_e_eff(w: Traced[float], h: Traced[float], t: Traced[float], er: Traced[float], ids: tuple[str, str, str, str] = ("w", "h", "t", "er")) -> Traced[float]:
    """Microstrip effective permittivity (thickness-corrected): Hammerstad & Jensen 1980."""
    r = microstrip(_v(w), _v(h), _v(t), _v(er))
    return _derived(r.e_eff, "calc.tline.microstrip.e_eff", ids, None, "e_eff = e_eff(u_r) (Z01(u_1)/Z01(u_r))^2 (Hammerstad & Jensen 1980; quasi-static)")


def stripline_z0(w: Traced[float], b: Traced[float], t: Traced[float], er: Traced[float], ids: tuple[str, str, str, str] = ("w", "b", "t", "er")) -> Traced[float]:
    """Centred stripline Z0 (ohm): Cohn 1954 (t = 0) or Wheeler 1978 (t > 0)."""
    z = stripline_impedance(_v(w), _v(b), _v(t), _v(er))
    note = "Z0 = eta0/(4 sqrt(er)) K(k)/K(k') (Cohn 1954, t = 0)" if _v(t) == 0 else "Z0 = eta0/(4 pi sqrt(er)) ln(1 + y(2y + sqrt((2y)^2 + 6.27))) (Wheeler 1978)"
    return _derived(z, "calc.tline.stripline.z0", ids, "ohm", note)


def _coupled(w: Traced, s: Traced, h: Traced, t: Traced, er: Traced) -> CoupledResult:
    return edge_coupled_microstrip(_v(w), _v(s), _v(h), _v(t), _v(er))


_KJ = "Kirschning & Jansen 1984 static, Jansen thickness widths"


def edge_coupled_z_even(w: Traced[float], s: Traced[float], h: Traced[float], t: Traced[float], er: Traced[float], ids: tuple[str, str, str, str, str] = ("w", "s", "h", "t", "er")) -> Traced[float]:
    """Even-mode impedance of an edge-coupled microstrip pair (ohm)."""
    return _derived(_coupled(w, s, h, t, er).z_even, "calc.tline.edge_coupled_microstrip.z_even", ids, "ohm", f"Z_even ({_KJ})")


def edge_coupled_z_odd(w: Traced[float], s: Traced[float], h: Traced[float], t: Traced[float], er: Traced[float], ids: tuple[str, str, str, str, str] = ("w", "s", "h", "t", "er")) -> Traced[float]:
    """Odd-mode impedance of an edge-coupled microstrip pair (ohm)."""
    return _derived(_coupled(w, s, h, t, er).z_odd, "calc.tline.edge_coupled_microstrip.z_odd", ids, "ohm", f"Z_odd ({_KJ})")


def edge_coupled_z_diff(w: Traced[float], s: Traced[float], h: Traced[float], t: Traced[float], er: Traced[float], ids: tuple[str, str, str, str, str] = ("w", "s", "h", "t", "er")) -> Traced[float]:
    """Differential impedance of an edge-coupled microstrip pair: Z_diff = 2 Z_odd (ohm)."""
    return _derived(_coupled(w, s, h, t, er).z_diff, "calc.tline.edge_coupled_microstrip.z_diff", ids, "ohm", f"Z_diff = 2 Z_odd ({_KJ})")


def edge_coupled_e_eff_even(w: Traced[float], s: Traced[float], h: Traced[float], t: Traced[float], er: Traced[float], ids: tuple[str, str, str, str, str] = ("w", "s", "h", "t", "er")) -> Traced[float]:
    """Even-mode effective permittivity of an edge-coupled microstrip pair."""
    return _derived(_coupled(w, s, h, t, er).e_eff_even, "calc.tline.edge_coupled_microstrip.e_eff_even", ids, None, f"e_eff,even ({_KJ})")


def edge_coupled_e_eff_odd(w: Traced[float], s: Traced[float], h: Traced[float], t: Traced[float], er: Traced[float], ids: tuple[str, str, str, str, str] = ("w", "s", "h", "t", "er")) -> Traced[float]:
    """Odd-mode effective permittivity of an edge-coupled microstrip pair."""
    return _derived(_coupled(w, s, h, t, er).e_eff_odd, "calc.tline.edge_coupled_microstrip.e_eff_odd", ids, None, f"e_eff,odd ({_KJ})")


def width_for_z0_microstrip(z0: Traced[float], h: Traced[float], t: Traced[float], er: Traced[float], ids: tuple[str, str, str, str] = ("z0", "h", "t", "er")) -> Traced[float]:
    """The microstrip width (mm) for a target Z0: bisection on Hammerstad & Jensen."""
    w = solve_microstrip_width(_v(z0), _v(h), _v(t), _v(er))
    return _derived(w, "calc.tline.width_for_z0.microstrip", ids, "mm", f"w with Z0(w) = Z0_target (bisection on calc.tline.microstrip.z0, bracket <= {SOLVE_REL_TOL:g} relative)")


def width_for_z0_stripline(z0: Traced[float], b: Traced[float], t: Traced[float], er: Traced[float], ids: tuple[str, str, str, str] = ("z0", "b", "t", "er")) -> Traced[float]:
    """The centred-stripline width (mm) for a target Z0: bisection on calc.tline.stripline.z0."""
    w = solve_stripline_width(_v(z0), _v(b), _v(t), _v(er))
    return _derived(w, "calc.tline.width_for_z0.stripline", ids, "mm", f"w with Z0(w) = Z0_target (bisection on calc.tline.stripline.z0, bracket <= {SOLVE_REL_TOL:g} relative)")


def width_for_z0(z0: Traced[float], h: Traced[float], t: Traced[float], er: Traced[float], kind: str, ids: tuple[str, str, str, str] | None = None) -> Traced[float]:
    """:func:`width_for_z0_microstrip` (``kind="microstrip"``, ``h`` the substrate height) or :func:`width_for_z0_stripline` (``"stripline"``, ``h`` the plane spacing b)."""
    if kind == "microstrip":
        return width_for_z0_microstrip(z0, h, t, er, ids or ("z0", "h", "t", "er"))
    if kind == "stripline":
        return width_for_z0_stripline(z0, h, t, er, ids or ("z0", "b", "t", "er"))
    raise ValueError(f"kind must be 'microstrip' or 'stripline', got {kind!r}")


def spacing_for_zdiff(z_diff: Traced[float], w: Traced[float], h: Traced[float], t: Traced[float], er: Traced[float], ids: tuple[str, str, str, str, str] = ("z_diff", "w", "h", "t", "er")) -> Traced[float]:
    """The gap (mm) of an edge-coupled microstrip pair of width ``w`` for a target Z_diff."""
    s = solve_coupled_spacing(_v(z_diff), _v(w), _v(h), _v(t), _v(er))
    return _derived(s, "calc.tline.edge_coupled_microstrip.s_for_zdiff", ids, "mm", f"s with Z_diff(s) = Z_diff_target (bisection on calc.tline.edge_coupled_microstrip.z_diff, bracket <= {SOLVE_REL_TOL:g} relative)")


def width_for_zdiff(z_diff: Traced[float], s: Traced[float], h: Traced[float], t: Traced[float], er: Traced[float], ids: tuple[str, str, str, str, str] = ("z_diff", "s", "h", "t", "er")) -> Traced[float]:
    """The width (mm) of an edge-coupled microstrip pair with gap ``s`` for a target Z_diff."""
    w = solve_coupled_width(_v(z_diff), _v(s), _v(h), _v(t), _v(er))
    return _derived(w, "calc.tline.edge_coupled_microstrip.w_for_zdiff", ids, "mm", f"w with Z_diff(w) = Z_diff_target (bisection on calc.tline.edge_coupled_microstrip.z_diff, bracket <= {SOLVE_REL_TOL:g} relative)")


def tpd(e_eff: Traced[float], ids: tuple[str] = ("e_eff",)) -> Traced[float]:
    """Propagation delay per length t_pd = sqrt(e_eff) / c0 (s/m)."""
    return _derived(propagation_delay(_v(e_eff)), "calc.tline.tpd", ids, "s/m", "t_pd = sqrt(e_eff) / c0")


def line_delay(length: Traced[float], t_pd: Traced[float], ids: tuple[str, str] = ("length", "t_pd")) -> Traced[float]:
    """Flight time of a line of ``length`` mm: t_d = l t_pd (s)."""
    return _derived(line_delay_s(_v(length), _v(t_pd)), "calc.tline.delay", ids, "s", "t_d = l * t_pd (l in mm / 1000)")


def critical_length(t_r: Traced[float], t_pd: Traced[float], fraction: Traced[float], ids: tuple[str, str, str] = ("t_r", "t_pd", "fraction")) -> Traced[float]:
    """The length (mm) above which a line is treated as a transmission line: l_crit = fraction t_r / t_pd."""
    return _derived(critical_length_mm(_v(t_r), _v(t_pd), _v(fraction)), "calc.tline.critical_length", ids, "mm", "l_crit = fraction * t_r / t_pd (mm; a rule, not a simulation)")


# --------------------------------------------------------------------------- the geometry a stackup gives a routed layer


@dataclass(frozen=True)
class LineGeometry:
    """What the calculators need for a track on ``layer``, read from the stackup, with the ids of the traced inputs.

    ``kind`` is ``"microstrip"``: an outer signal layer over the adjacent
    plane layer ``reference_layer`` (net ``reference_net``), ``h`` the
    dielectric between them, ``t`` the layer's copper, ``er`` that
    dielectric's permittivity. ``notes`` name what the closed form leaves
    out for this geometry (the solder mask, a Dk frequency not stated).
    """

    layer: str
    kind: str
    h: Traced
    h_id: str
    t: Traced
    t_id: str
    er: Traced
    er_id: str
    reference_layer: str
    reference_net: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ids(self) -> tuple[str, str, str]:
        """``(h_id, t_id, er_id)``: the ids a calculator records for the stackup inputs, after the width's id."""
        return (self.h_id, self.t_id, self.er_id)


#: the reason every impedance check gives when the layer has no adjacent plane
NO_REFERENCE_PLANE = "impedance is undefined without a reference plane"
NO_STACKUP = "no stackup"


def line_geometry(stackup, layer: str, prefix: str = "pcb.stackup") -> tuple[LineGeometry | None, str | None]:
    """``(geometry, None)`` for a signal track on ``layer`` of ``stackup`` (an :class:`ai_eda.ir.Stackup`), or ``(None, reason)``.

    Reasons: no stackup; a layer the stack does not have; a plane layer
    (nothing is routed on it); an outer layer whose adjacent copper is not a
    plane (:data:`NO_REFERENCE_PLANE`); an inner signal layer (a stripline
    needs the plane spacing b, which this version does not derive from a
    stack - no inner layer is routed).
    """
    if stackup is None:
        return None, NO_STACKUP
    names = stackup.copper_names()
    if layer not in names:
        return None, f"layer {layer!r} is not in the stackup {names}"
    i = stackup.index(layer)
    copper = stackup.copper[i]
    if copper.plane_net is not None:
        return None, f"{layer} is a plane layer ({copper.plane_net.value}); no signal is routed on it"
    if 0 < i < len(names) - 1:
        return None, f"{layer} is an inner signal layer: stripline geometry (the plane spacing b) is not derived from a stackup by this version"
    j = 1 if i == 0 else i - 1
    neighbour = stackup.copper[j]
    if neighbour.plane_net is None:
        return None, f"{NO_REFERENCE_PLANE}: the copper next to {layer} ({neighbour.name}) is not a plane layer"
    d_index = min(i, j)
    d = stackup.dielectrics[d_index]
    notes: list[str] = []
    if stackup.solder_mask is not None:
        notes.append("the closed form is for an uncoated microstrip: the solder mask over the track is not modelled (it lowers Z0)")
    if d.er_frequency_hz is None:
        notes.append(f"the frequency of the dielectric's er ({float(d.er.value):g}) is not recorded")
    else:
        notes.append(f"er {float(d.er.value):g} as stated at {float(d.er_frequency_hz.value):g} Hz (dispersion not modelled)")
    return LineGeometry(
        layer=layer, kind="microstrip",
        h=d.thickness_mm, h_id=f"{prefix}.dielectrics[{d_index}].thickness_mm",
        t=copper.thickness_um, t_id=f"{prefix}.copper[{layer}].thickness_um",
        er=d.er, er_id=f"{prefix}.dielectrics[{d_index}].er",
        reference_layer=neighbour.name, reference_net=str(neighbour.plane_net.value), notes=tuple(notes),
    ), None


__all__ = [
    "C0",
    "COUPLED_ER_RANGE",
    "COUPLED_G_RANGE",
    "COUPLED_U_RANGE",
    "ETA0",
    "LUMPED_FRACTION",
    "MICROSTRIP_ER_RANGE",
    "MICROSTRIP_U_RANGE",
    "MU0",
    "NO_REFERENCE_PLANE",
    "NO_STACKUP",
    "ROUND_TRIP_FRACTION",
    "SOLVE_REL_TOL",
    "STRIPLINE_TB_MAX",
    "STRIPLINE_WB_RANGE",
    "CoupledResult",
    "LineGeometry",
    "MicrostripResult",
    "TLineRangeError",
    "critical_length",
    "critical_length_mm",
    "edge_coupled_e_eff_even",
    "edge_coupled_e_eff_odd",
    "edge_coupled_microstrip",
    "edge_coupled_z_diff",
    "edge_coupled_z_even",
    "edge_coupled_z_odd",
    "line_delay",
    "line_delay_s",
    "line_geometry",
    "microstrip",
    "microstrip_e_eff",
    "microstrip_z0",
    "propagation_delay",
    "solve_coupled_spacing",
    "solve_coupled_width",
    "solve_microstrip_width",
    "solve_stripline_width",
    "spacing_for_zdiff",
    "stripline_impedance",
    "stripline_z0",
    "tpd",
    "width_for_z0",
    "width_for_z0_microstrip",
    "width_for_z0_stripline",
    "width_for_zdiff",
]
