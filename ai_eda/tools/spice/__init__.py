"""SPICE simulation interface.

The system never "predicts" a simulation result. A :class:`SpiceResult`
exists only if a real engine ran a real netlist file, and it records the
engine version, the hash of the file that ran, the exact analysis command,
every vector of the plot and the rawfile the engine wrote as evidence.

Engines:

* :class:`NgspiceShared` - KiCad's bundled ``ngspice.dll`` through ctypes
  (:mod:`ai_eda.tools.spice.ngspice_shared`); the one used on this machine.
* :class:`NgspiceRunner` - an ``ngspice`` batch binary on PATH; optional.

:mod:`ai_eda.tools.spice.rawfile` reads the ASCII and binary rawfiles either
engine writes; :mod:`ai_eda.tools.spice.measure` (``rising_edge_frequency``)
measures a frequency from a transient vector's samples (pure, no engine).
:mod:`ai_eda.tools.spice.stage` (``run_spice_for``) drives an
engine from an IR's simulation setup and judges the expectations; it imports
``ai_eda.ir`` and is therefore not re-exported here (``ai_eda.ir.simulation``
imports this package for :class:`SpiceAnalysis`).
"""

from ai_eda.tools.spice import rawfile
from ai_eda.tools.spice.measure import EdgeFrequency, rising_edge_frequency
from ai_eda.tools.spice.ngspice_shared import NgspiceShared, find_ngspice_dll, validate_deck
from ai_eda.tools.spice.runner import (
    Interpolation,
    NgspiceRunner,
    SpiceAnalysis,
    SpiceResult,
    SpiceRunner,
    normalise_command,
    result_from_rawfile,
)

__all__ = [
    "EdgeFrequency",
    "Interpolation",
    "NgspiceRunner",
    "NgspiceShared",
    "SpiceAnalysis",
    "SpiceResult",
    "SpiceRunner",
    "find_ngspice_dll",
    "normalise_command",
    "rawfile",
    "result_from_rawfile",
    "rising_edge_frequency",
    "validate_deck",
]
