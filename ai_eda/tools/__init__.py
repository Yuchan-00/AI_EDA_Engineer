"""Deterministic tools.

Everything in this package is a *fact producer*: calculators, SPICE runners,
KiCad CLI wrappers, manufacturing-file checkers. None of them call an LLM.
Their outputs carry ``derived`` provenance with the tool id and version.
"""
