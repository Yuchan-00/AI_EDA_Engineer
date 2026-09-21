"""AI EDA ENGINEER.

Verification-first AI electronic design engineering system.

Design principle: the LLM never decides design truth. The Universal Circuit IR
(:mod:`ai_eda.ir`) is the single source of design data, every important value
carries provenance, and every claim of correctness comes from a deterministic
tool (calculator, SPICE, KiCad ERC/DRC, manufacturing checks) or an
independent review - never from model output alone.
"""

__version__ = "0.0.1"
