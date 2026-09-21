"""IR -> SPICE netlist compiler (scaffold).

Needs a per-component SPICE model mapping (R/C/L are trivial; ICs need a
vendor model with authoritative provenance). Until the mapping exists the
compiler raises rather than emitting a netlist with invented models.
"""

from __future__ import annotations

from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR
from ai_eda.compilers.base import CompileContext, Compiler


class SpiceNetlistCompiler(Compiler):
    id = "compiler.spice"
    kind = ArtifactKind.SPICE_NETLIST

    def compile(self, ir: CircuitIR, ctx: CompileContext) -> ArtifactRef:
        raise NotImplementedError("SPICE model mapping not implemented yet")
