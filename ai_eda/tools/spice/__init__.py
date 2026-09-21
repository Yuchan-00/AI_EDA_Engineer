"""SPICE simulation interface.

The system never "predicts" a simulation result. A :class:`SpiceResult`
exists only if a real engine ran a real netlist, and it records the engine
version and the netlist hash it ran on.
"""

from ai_eda.tools.spice.runner import NgspiceRunner, SpiceAnalysis, SpiceResult, SpiceRunner

__all__ = ["NgspiceRunner", "SpiceAnalysis", "SpiceResult", "SpiceRunner"]
