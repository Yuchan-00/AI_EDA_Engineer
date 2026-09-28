"""Simulation setup inside the IR: SPICE bindings, stimuli, analyses, expectations.

Invariant: **the LLM never decides what is simulated or what the answer should
be.** Everything here is ``Traced`` so :mod:`ai_eda.compilers.spice` can refuse
``llm_generated`` values, params and model cards (a model card must even be
authoritative or user-supplied) and so the reviewer can see which provenance
kinds reached ngspice. The compiler also never guesses: a component without a
:class:`SpiceBinding` is a ``CompileError``; a part that has no SPICE meaning
(a connector, a test point) is bound with ``exclude=True`` and a reason.

Analyses are *not* written into the netlist (no ``.control`` block, no
``.tran``/``.ac`` cards): :func:`ai_eda.compilers.spice.analysis_command`
turns an :class:`AnalysisSpec` into the interactive command the runner issues,
so one netlist serves every analysis and the command text is a pure function
of the IR.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from ai_eda.ir.provenance import Provenance, Traced, drop_empty_in_design_view
from ai_eda.tools.spice.runner import SpiceAnalysis


class SpiceDevice(StrEnum):
    """SPICE element letter: it prefixes the element name when the IR ref does not already start with it."""

    R = "R"
    C = "C"
    L = "L"
    V = "V"
    I = "I"
    D = "D"
    Q = "Q"
    M = "M"
    X = "X"  # subcircuit instance
    T = "T"  # lossless transmission line: port 1 (+, -), port 2 (+, -); params z0 (ohm) and td (s)


#: devices whose element line carries a plain value (ohm / F / H / V / A)
VALUE_DEVICES: frozenset[SpiceDevice] = frozenset({SpiceDevice.R, SpiceDevice.C, SpiceDevice.L, SpiceDevice.V, SpiceDevice.I})
#: devices whose element line names a ``.model`` (D, Q, M) or ``.subckt`` (X)
MODEL_DEVICES: frozenset[SpiceDevice] = frozenset({SpiceDevice.D, SpiceDevice.Q, SpiceDevice.M, SpiceDevice.X})
#: devices whose element line carries only named params, each one required (no value, no model): the lossless line
#: ``T<ref> p1+ p1- p2+ p2- td=<s> z0=<ohm>`` - ngspice-42 simulates a T line without ``td`` with a delay of its own
#: choosing and no error (measured), so the compiler never writes one without both
PARAM_DEVICES: frozenset[SpiceDevice] = frozenset({SpiceDevice.T})
#: the params each :data:`PARAM_DEVICES` device requires, exactly (sorted: the order the netlist writes them), with their units
DEVICE_PARAMS: dict[SpiceDevice, dict[str, str]] = {SpiceDevice.T: {"td": "s", "z0": "ohm"}}
#: devices with exactly two nodes
TWO_TERMINAL_DEVICES: frozenset[SpiceDevice] = frozenset(
    {SpiceDevice.R, SpiceDevice.C, SpiceDevice.L, SpiceDevice.V, SpiceDevice.I, SpiceDevice.D}
)
#: node counts ngspice accepts per device (``None`` = one or more, the subcircuit header decides)
NODE_COUNTS: dict[SpiceDevice, tuple[int, ...] | None] = {
    SpiceDevice.R: (2,),
    SpiceDevice.C: (2,),
    SpiceDevice.L: (2,),
    SpiceDevice.V: (2,),
    SpiceDevice.I: (2,),
    SpiceDevice.D: (2,),
    SpiceDevice.Q: (3, 4),  # collector base emitter [substrate]
    SpiceDevice.M: (4,),  # drain gate source bulk
    SpiceDevice.X: None,
    SpiceDevice.T: (4,),  # port 1 +, port 1 -, port 2 +, port 2 -
}

#: unit the compiler expects on ``SpiceBinding.value`` per value device (informational; the value itself is SI)
VALUE_UNITS: dict[SpiceDevice, str] = {
    SpiceDevice.R: "ohm",
    SpiceDevice.C: "F",
    SpiceDevice.L: "H",
    SpiceDevice.V: "V",
    SpiceDevice.I: "A",
}
#: the ``Component.electrical`` key that holds the same physical quantity as ``SpiceBinding.value``; when
#: the component has it, the compiler requires the two numbers to agree (the simulated part is the shipped part)
ELECTRICAL_KEYS: dict[SpiceDevice, str] = {
    SpiceDevice.R: "resistance",
    SpiceDevice.C: "capacitance",
    SpiceDevice.L: "inductance",
}


class SpiceBinding(BaseModel):
    """How one IR component appears in the SPICE netlist.

    * ``device`` - the element type; ``None`` is only legal together with ``exclude``.
    * ``exclude`` / ``exclude_reason`` - the component is left out of the netlist
      (connectors, mounting holes, test points). The reason is reported.
    * ``value`` - ohm / F / H / V / A for R, C, L, V, I (SI, not a suffix string).
      For R / C / L it must equal ``Component.electrical["resistance" |
      "capacitance" | "inductance"]`` when that entry exists (the value the
      BOM ships); the compiler refuses a netlist that would simulate a
      different part than the one built.
    * ``model_name`` - the ``.model`` / ``.subckt`` name referenced by D, Q, M, X
      (or a semiconductor R/C/L model).
    * ``model_card`` - the verbatim ``.model`` / ``.subckt`` text that defines
      ``model_name``; it must carry authoritative or user provenance. Several
      components may name the same model; one card per model name is enough.
    * ``pin_order`` - IR pin numbers in SPICE node order (default ``["1", "2"]``
      for two-terminal parts). Must be a permutation of (a subset of) the
      component's pins; every listed pin must be in a net.
    * ``ignored_pins`` - pin number -> reason, for a pin that *is* wired into a
      net but that the SPICE element does not use (a potentiometer's wiper on
      a 2-node ``R``, a transistor's substrate on a 3-node ``Q``). The
      schematic connects such a pin and the netlist does not, so the compiler
      refuses a connected pin that is neither in ``pin_order`` nor listed here,
      and reports the listed ones.
    * ``params`` - extra ``name=value`` tokens (``ic``, ``area``, ``m``, subckt params);
      for a lossless line (``device=T``, four pins in ``pin_order``: port 1
      +/-, port 2 +/-) exactly ``z0`` (ohm) and ``td`` (s), both positive
      (:data:`DEVICE_PARAMS`), and no value or model.
    * ``provenance`` - who decided this binding (device, pin order, exclusion). An
      ``llm_generated`` binding is refused by the compiler like an
      ``llm_generated`` value; an ``assumption`` is reported.
    """

    device: SpiceDevice | None = None
    exclude: bool = False
    exclude_reason: str = ""
    value: Traced[float] | None = None
    model_name: str | None = None
    model_card: Traced[str] | None = None
    pin_order: list[str] = Field(default_factory=lambda: ["1", "2"])
    ignored_pins: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Traced] = Field(default_factory=dict)
    #: why this binding was chosen (which datasheet / library / decision)
    provenance: Provenance


class StimulusKind(StrEnum):
    """The waveform of a :class:`Stimulus` (ngspice's own independent-source functions; the compiler fills no default).

    * ``DC`` - ``value`` only.
    * ``PULSE`` / ``SINE`` / ``PWL`` - ngspice's ``PULSE(...)`` / ``SINE(...)`` / ``PWL(...)``.
    * ``AM`` - ngspice's amplitude-modulated source ``AM(VA VO MF FC TD)``:
      v(t) = VA (VO + sin(2 pi MF (t - TD))) sin(2 pi FC (t - TD)) for t >= TD, 0 before;
      the modulation depth is m = 1 / VO (the envelope swings between VA (VO - 1)
      and VA (VO + 1)).
    * ``SFFM`` - ngspice's single-frequency FM source ``SFFM(VO VA FC MDI FS)``:
      v(t) = VO + VA sin(2 pi FC t + MDI sin(2 pi FS t)); FC is the carrier, MDI
      the modulation index (peak deviation / FS), FS the modulating frequency.

    Measured on ngspice-42 (Ubuntu libngspice0, 2026-09-28, through
    :class:`~ai_eda.tools.spice.NgspiceShared`, and in batch): ``AM(1 2 1k 100k 0)``
    peaks at 2.99941 V = VA (VO + 1) (``AM(0.5 2 100k 900meg 0)`` at 1.4998 V);
    ``AM(1 2 1k 20k 0.5m)`` is exactly 0 V before TD; VO = 0 is honoured
    (DSB-SC); ``SFFM(0 1 1k 0 10k)`` is the plain 1 kHz sine (MDI = 0 honoured)
    and ``SFFM(0 1 1k 5 10k)`` crosses zero exactly where
    sin(2 pi 1k t + 5 sin(2 pi 10k t)) does (10.54, 51.02, 86.80 us). But
    **ngspice-42 silently replaces some zero parameters by defaults of its own**:
    ``AM(1 2 0 20k 0)`` (MF = 0) peaks at 3.0 V, not the unmodulated 2.0 V, and
    ``SFFM(0 1 1k 5 0)`` (FS = 0) is not the 1 kHz sine it would be. So the
    compiler refuses a non-positive ``mf`` / ``fc`` (AM) and ``fc`` / ``fs``
    (SFFM), requires every parameter finite and ``td >= 0``, and accepts no
    optional trailing parameter (ngspice's PHASEM / PHASEC / PHASES are not
    measured here).
    """

    DC = "dc"
    PULSE = "pulse"
    SINE = "sine"
    PWL = "pwl"
    AM = "am"
    SFFM = "sffm"


#: ordered ngspice parameters of each transient stimulus kind (all required: the compiler does not fill defaults)
STIMULUS_PARAMS: dict[StimulusKind, tuple[str, ...]] = {
    StimulusKind.DC: (),
    StimulusKind.PULSE: ("v1", "v2", "td", "tr", "tf", "pw", "per"),
    StimulusKind.SINE: ("vo", "va", "freq"),
    StimulusKind.PWL: ("points",),
    StimulusKind.AM: ("va", "vo", "mf", "fc", "td"),
    StimulusKind.SFFM: ("vo", "va", "fc", "mdi", "fs"),
}
#: optional trailing SINE parameters, only accepted as a contiguous prefix of this order
SINE_OPTIONAL_PARAMS: tuple[str, ...] = ("td", "theta", "phase")
#: parameters that must be > 0: ngspice-42 replaces a zero one by a default of its own and simulates another waveform
#: (measured, :class:`StimulusKind`), so the compiler refuses it instead
STIMULUS_POSITIVE_PARAMS: dict[StimulusKind, tuple[str, ...]] = {
    StimulusKind.AM: ("mf", "fc"),
    StimulusKind.SFFM: ("fc", "fs"),
}
#: parameters that must be >= 0 (a delay)
STIMULUS_NON_NEGATIVE_PARAMS: dict[StimulusKind, tuple[str, ...]] = {
    StimulusKind.AM: ("td",),
}
#: ``ac`` (small-signal magnitude) may be added to any stimulus; an AC analysis needs at least one
AC_MAGNITUDE_PARAM = "ac"


class Stimulus(BaseModel):
    """An independent source the *simulation* adds between two IR nets.

    It becomes ``V<id>`` / ``I<id>`` in the netlist (``VVIN`` for id ``VIN``),
    connected from ``net`` (+) to ``reference_net`` (-). ``value`` is the DC
    level for :attr:`StimulusKind.DC`; the other kinds take their numbers from
    ``params`` (see :data:`STIMULUS_PARAMS`; PWL takes ``points`` as a traced
    list of ``[t, v]`` pairs).
    """

    id: str
    source: Literal["voltage", "current"]
    net: str
    reference_net: str
    kind: StimulusKind
    value: Traced[float] | None = None
    params: dict[str, Traced] = Field(default_factory=dict)
    provenance: Provenance
    serves_requirements: list[str] = Field(default_factory=list)


#: parameter names each analysis kind accepts (``dc.source`` is a stimulus id; tran ``uic`` is a bool: skip the
#: operating point and start the transient from the elements' ``ic`` params - ngspice's ``uic`` keyword)
ANALYSIS_PARAMS: dict[SpiceAnalysis, tuple[str, ...]] = {
    SpiceAnalysis.OP: (),
    SpiceAnalysis.DC: ("source", "start", "stop", "step"),
    SpiceAnalysis.TRAN: ("step", "stop", "start", "uic"),
    SpiceAnalysis.AC: ("variation", "points", "fstart", "fstop"),
}
ANALYSIS_OPTIONAL_PARAMS: dict[SpiceAnalysis, tuple[str, ...]] = {
    SpiceAnalysis.OP: (),
    SpiceAnalysis.DC: (),
    SpiceAnalysis.TRAN: ("start", "uic"),
    SpiceAnalysis.AC: (),
}
AC_VARIATIONS: tuple[str, ...] = ("dec", "oct", "lin")


class AnalysisSpec(BaseModel):
    """One ngspice analysis; ``kind`` reuses :class:`ai_eda.tools.spice.SpiceAnalysis`.

    ``params``: tran ``step``, ``stop``, [``start``], [``uic``] (a bool: skip
    the operating point, start from the elements' ``ic`` params); dc
    ``source`` (stimulus id), ``start``, ``stop``, ``step``; ac ``variation``
    (``dec`` | ``oct`` | ``lin``), ``points``, ``fstart``, ``fstop``; op none.
    """

    id: str
    kind: SpiceAnalysis
    params: dict[str, Traced] = Field(default_factory=dict)
    provenance: Provenance


class Reduce(StrEnum):
    """How a result vector is reduced to the one number compared with ``nominal``."""

    VALUE = "value"  # op / single point
    #: value at ``at`` on the sweep axis: an exact grid hit, else an interpolation between the two
    #: neighbouring samples (linear; in log frequency, and log magnitude, for an ac sweep) that is
    #: judged only when those two samples themselves lie within the tolerance - otherwise UNRESOLVED
    AT = "at"
    FINAL = "final"
    MAX = "max"
    MIN = "min"
    #: the mean frequency of the rising mid-level crossings of a transient vector over the saved window,
    #: f = (N - 1) / (t_N - t_1) for N >= 3 crossings (:func:`ai_eda.tools.spice.measure.rising_edge_frequency`).
    #: Crossings are detected with hysteresis: armed once a sample is at or below vmin + 0.25 * swing, counted at the
    #: first later sample at or above vmin + 0.75 * swing, the crossing time interpolated linearly at the mid level
    #: vmin + 0.5 * swing between the two samples that bracket it. Fewer than 3 crossings, a swing within the engine's
    #: own resolution (ngspice's ``reltol`` * level + ``vntol`` / ``abstol``: numerical ripple is not an oscillation),
    #: non-finite samples or a time scale that steps backwards give no number ("no oscillation detected"), which the
    #: stage reports as FAIL, never PASS. Only on a tran analysis (the compiler refuses it elsewhere).
    FREQUENCY = "frequency"
    #: a level in dB at one frequency of an ac sweep: 20 log10(|vector(at)| / R), R = |reference_vector(at)| or
    #: ``params["ref"]`` (> 0, the vector's own unit). Both magnitudes are read with :meth:`SpiceResult.interpolate`
    #: at ``at`` from the same two samples, and the dB bracket of those samples is judged like ``AT`` (UNRESOLVED
    #: when it straddles the tolerance). Output unit ``dB``; only on an ac analysis, the vector a magnitude.
    DB_AT = "db_at"
    #: the RMS of a tran vector over the window ``[t_start, t_stop]`` (``params``): the exact integral of the
    #: piecewise-linear interpolant of the samples ngspice wrote (:func:`ai_eda.tools.spice.measure.window_rms`).
    #: ``params["f_max"]`` (Hz, required) is the highest frequency the RMS must include: a grid coarser than
    #: 1 / (20 f_max) gives no number (on a coarse grid the interpolant's RMS reads low, and aliases at one sample
    #: per period), and the verdict is judged on [RMS, RMS / (1 - b)] with the read-low bound b of the largest step.
    #: Output unit: the vector's (V or A).
    RMS = "rms"
    #: 20 log10(RMS(vector) / RMS(reference_vector)) over the window, or against ``params["ref"]`` (an RMS level in
    #: the vector's unit), with the same ``f_max`` guard and bias bracket. Output unit ``dB``; a zero reference RMS gives no number.
    DB_RMS = "db_rms"
    #: the level of harmonic ``k`` (an integer >= 2) of ``f0`` relative to the fundamental, in dBc, from the exact
    #: Fourier integral of the piecewise-linear interpolant over the window, which must hold at least two whole
    #: periods of ``f0`` (:func:`ai_eda.tools.spice.measure.harmonic_level`); a grid coarser than 1 / (20 k f0) or a
    #: waveform that does not repeat at ``f0`` (its measured fundamental is another frequency, an unsettled start, a
    #: modulation: the periodicity residual above 2 % of the fundamental) gives no number, and the verdict is judged
    #: on the bracket of the grid attenuation and the leakage bound, never on the bare number.
    HARMONIC_DBC = "harmonic_dbc"
    #: the AM modulation depth 100 (A_max - A_min) / (A_max + A_min) in **percent**, from the per-carrier-period
    #: amplitude (max - min) / 2 over the window (:func:`ai_eda.tools.spice.measure.am_depth`); ``f_carrier`` >= 20
    #: ``f_mod``, a window of at least one modulation period and a grid of at most 1 / (16 f_carrier); judged on the
    #: bracket of the crest and envelope-smear bias bounds, never on the bare number.
    AM_DEPTH = "am_depth"


#: ``Expectation.params`` each reduction takes: (required, optional). Every other reduction takes none.
REDUCE_PARAMS: dict[Reduce, tuple[tuple[str, ...], tuple[str, ...]]] = {
    Reduce.VALUE: ((), ()),
    Reduce.AT: ((), ()),
    Reduce.FINAL: ((), ()),
    Reduce.MAX: ((), ()),
    Reduce.MIN: ((), ()),
    Reduce.FREQUENCY: ((), ()),
    Reduce.DB_AT: ((), ("ref",)),
    Reduce.RMS: (("t_start", "t_stop", "f_max"), ()),
    Reduce.DB_RMS: (("t_start", "t_stop", "f_max"), ("ref",)),
    Reduce.HARMONIC_DBC: (("f0", "k", "t_start", "t_stop"), ()),
    Reduce.AM_DEPTH: (("f_carrier", "f_mod", "t_start", "t_stop"), ()),
}
#: reductions whose number is a level in dB: a relative tolerance on a logarithm is no tolerance, so they need ``tol_abs``
DB_REDUCES: frozenset[Reduce] = frozenset({Reduce.DB_AT, Reduce.DB_RMS, Reduce.HARMONIC_DBC})
#: reductions that read a reference (``reference_vector`` or ``params["ref"]``, exactly one)
REFERENCE_REDUCES: frozenset[Reduce] = frozenset({Reduce.DB_AT, Reduce.DB_RMS})
#: reductions over a time window ``[t_start, t_stop]`` of a tran analysis
WINDOW_REDUCES: frozenset[Reduce] = frozenset({Reduce.RMS, Reduce.DB_RMS, Reduce.HARMONIC_DBC, Reduce.AM_DEPTH})
#: the unit of the number a reduction gives, when it is fixed (``RMS`` gives the vector's own unit)
REDUCE_UNITS: dict[Reduce, str] = {Reduce.DB_AT: "dB", Reduce.DB_RMS: "dB", Reduce.HARMONIC_DBC: "dBc", Reduce.AM_DEPTH: "percent"}
#: the unit a reduction param must carry when its ``Traced.unit`` is set (``k`` is a plain count and carries none;
#: ``ref`` is in the vector's own unit, V for ``v()`` and A for ``i()``)
REDUCE_PARAM_UNITS: dict[str, str | None] = {"t_start": "s", "t_stop": "s", "f0": "Hz", "f_carrier": "Hz", "f_mod": "Hz", "f_max": "Hz", "k": None}


class Expectation(BaseModel):
    """What a real simulation result must show for a requirement to count as verified.

    ``vector`` uses a restricted grammar that the compiler parses and
    validates against the IR; it is never handed to ngspice as an expression:

    * ``v(<NET>)`` - node voltage (the magnitude for an ac analysis),
    * ``i(<STIMULUS_ID or REF>)`` - branch current of a voltage source,
    * ``vp(<NET>)`` / ``ip(...)`` - phase in degrees, ``vr()`` / ``ir()`` real
      part, ``vi()`` / ``ii()`` imaginary part - ac analyses only.

    ngspice's ``vdb()`` is not a vector here: a level in decibels is a
    reduction (``reduce=db_at`` on an ac magnitude, ``db_rms`` /
    ``harmonic_dbc`` on a tran window - :data:`DB_REDUCES`), computed by this
    project from the magnitudes ngspice wrote, with the reference in
    ``reference_vector`` (the same grammar as ``vector``, never a complex
    part) or ``params["ref"]``. A dB expectation needs ``tol_abs`` (a
    relative tolerance on a logarithm is no tolerance) and, when
    ``nominal.unit`` is set, the reduction's unit (:data:`REDUCE_UNITS`:
    ``dB``, ``dBc``; ``percent`` for ``am_depth``).

    ``params`` are the reduction's own numbers (:data:`REDUCE_PARAMS`: the
    window ``t_start`` / ``t_stop`` in s, ``f0`` / ``f_carrier`` / ``f_mod``
    / ``f_max`` in Hz, the harmonic number ``k``, the reference level ``ref``), each
    ``Traced`` like every number the compiler accepts; a reduction that takes
    none refuses any. ``params`` and ``reference_vector`` are left out of the
    design view while empty, so an IR saved before they existed keeps its
    hash.

    ``requirement_id`` names the requirement this expectation verifies. When
    that requirement carries a numeric ``value``, the reviewer requires
    ``nominal`` to agree with it (same unit, within this expectation's
    tolerance): an expectation cannot claim to verify a 6 V requirement with a
    4 V nominal. ``nominal == 0`` needs ``tol_abs`` (a relative tolerance on
    zero is no tolerance). Expectations are judged at the components' nominal
    values and one temperature: no tolerance corners are simulated, and every
    result says so (``details["conditions"]``).
    """

    id: str
    analysis_id: str
    vector: str
    reduce: Reduce = Reduce.VALUE
    at: Traced[float] | None = None
    nominal: Traced[float]
    tol_abs: Traced[float] | None = None
    #: fraction of ``nominal`` (0.05 = 5 %)
    tol_rel: Traced[float] | None = None
    requirement_id: str | None = None
    #: the reduction's own numbers (:data:`REDUCE_PARAMS`)
    params: dict[str, Traced] = Field(default_factory=dict)
    #: the reference of a ``db_at`` / ``db_rms`` level (the vector grammar above; a magnitude, never a complex part)
    reference_vector: str | None = None
    provenance: Provenance

    _design = drop_empty_in_design_view("params", "reference_vector")


class SimulationSetup(BaseModel):
    stimuli: list[Stimulus] = Field(default_factory=list)
    analyses: list[AnalysisSpec] = Field(default_factory=list)
    expectations: list[Expectation] = Field(default_factory=list)
    temperature_c: Traced[float] | None = None

    def stimulus(self, id: str) -> Stimulus | None:
        for s in self.stimuli:
            if s.id == id:
                return s
        return None

    def analysis(self, id: str) -> AnalysisSpec | None:
        for a in self.analyses:
            if a.id == id:
                return a
        return None


__all__ = [
    "AC_MAGNITUDE_PARAM",
    "AC_VARIATIONS",
    "ANALYSIS_OPTIONAL_PARAMS",
    "ANALYSIS_PARAMS",
    "AnalysisSpec",
    "DB_REDUCES",
    "DEVICE_PARAMS",
    "ELECTRICAL_KEYS",
    "Expectation",
    "MODEL_DEVICES",
    "NODE_COUNTS",
    "PARAM_DEVICES",
    "REDUCE_PARAMS",
    "REDUCE_PARAM_UNITS",
    "REDUCE_UNITS",
    "REFERENCE_REDUCES",
    "Reduce",
    "SINE_OPTIONAL_PARAMS",
    "STIMULUS_NON_NEGATIVE_PARAMS",
    "STIMULUS_PARAMS",
    "STIMULUS_POSITIVE_PARAMS",
    "SimulationSetup",
    "SpiceBinding",
    "SpiceDevice",
    "Stimulus",
    "StimulusKind",
    "TWO_TERMINAL_DEVICES",
    "VALUE_DEVICES",
    "VALUE_UNITS",
    "WINDOW_REDUCES",
]
