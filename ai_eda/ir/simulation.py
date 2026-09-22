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

from ai_eda.ir.provenance import Provenance, Traced
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


#: devices whose element line carries a plain value (ohm / F / H / V / A)
VALUE_DEVICES: frozenset[SpiceDevice] = frozenset({SpiceDevice.R, SpiceDevice.C, SpiceDevice.L, SpiceDevice.V, SpiceDevice.I})
#: devices whose element line names a ``.model`` (D, Q, M) or ``.subckt`` (X)
MODEL_DEVICES: frozenset[SpiceDevice] = frozenset({SpiceDevice.D, SpiceDevice.Q, SpiceDevice.M, SpiceDevice.X})
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
    * ``params`` - extra ``name=value`` tokens (``ic``, ``area``, ``m``, subckt params).
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
    DC = "dc"
    PULSE = "pulse"
    SINE = "sine"
    PWL = "pwl"


#: ordered ngspice parameters of each transient stimulus kind (all required: the compiler does not fill defaults)
STIMULUS_PARAMS: dict[StimulusKind, tuple[str, ...]] = {
    StimulusKind.DC: (),
    StimulusKind.PULSE: ("v1", "v2", "td", "tr", "tf", "pw", "per"),
    StimulusKind.SINE: ("vo", "va", "freq"),
    StimulusKind.PWL: ("points",),
}
#: optional trailing SINE parameters, only accepted as a contiguous prefix of this order
SINE_OPTIONAL_PARAMS: tuple[str, ...] = ("td", "theta", "phase")
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


#: parameter names each analysis kind accepts (``dc.source`` is a stimulus id)
ANALYSIS_PARAMS: dict[SpiceAnalysis, tuple[str, ...]] = {
    SpiceAnalysis.OP: (),
    SpiceAnalysis.DC: ("source", "start", "stop", "step"),
    SpiceAnalysis.TRAN: ("step", "stop", "start"),
    SpiceAnalysis.AC: ("variation", "points", "fstart", "fstop"),
}
ANALYSIS_OPTIONAL_PARAMS: dict[SpiceAnalysis, tuple[str, ...]] = {
    SpiceAnalysis.OP: (),
    SpiceAnalysis.DC: (),
    SpiceAnalysis.TRAN: ("start",),
    SpiceAnalysis.AC: (),
}
AC_VARIATIONS: tuple[str, ...] = ("dec", "oct", "lin")


class AnalysisSpec(BaseModel):
    """One ngspice analysis; ``kind`` reuses :class:`ai_eda.tools.spice.SpiceAnalysis`.

    ``params``: tran ``step``, ``stop``, [``start``]; dc ``source`` (stimulus id),
    ``start``, ``stop``, ``step``; ac ``variation`` (``dec`` | ``oct`` | ``lin``),
    ``points``, ``fstart``, ``fstop``; op none.
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


class Expectation(BaseModel):
    """What a real simulation result must show for a requirement to count as verified.

    ``vector`` uses a restricted grammar that the compiler parses and
    validates against the IR; it is never handed to ngspice as an expression:

    * ``v(<NET>)`` - node voltage (the magnitude for an ac analysis),
    * ``i(<STIMULUS_ID or REF>)`` - branch current of a voltage source,
    * ``vp(<NET>)`` / ``ip(...)`` - phase in degrees, ``vr()`` / ``ir()`` real
      part, ``vi()`` / ``ii()`` imaginary part - ac analyses only.

    Decibels (``vdb``) are not expressible: put the magnitude ratio in
    ``nominal`` instead.

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
    provenance: Provenance


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
    "ELECTRICAL_KEYS",
    "Expectation",
    "MODEL_DEVICES",
    "NODE_COUNTS",
    "Reduce",
    "SINE_OPTIONAL_PARAMS",
    "STIMULUS_PARAMS",
    "SimulationSetup",
    "SpiceBinding",
    "SpiceDevice",
    "Stimulus",
    "StimulusKind",
    "TWO_TERMINAL_DEVICES",
    "VALUE_DEVICES",
    "VALUE_UNITS",
]
