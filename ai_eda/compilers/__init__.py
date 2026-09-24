"""IR -> artifact compilers.

Every compiler is a pure function of the IR (plus verified library data).
Because of that, "regenerate from IR" is always a valid repair for a broken
artifact, and the generated file records the IR hash it came from.
"""

from ai_eda.compilers.base import Compiler, CompileContext
from ai_eda.compilers.schematic import SchematicCompiler
from ai_eda.compilers.pcb import PCBCompiler
from ai_eda.compilers.bom import BOMCompiler, CPLCompiler, TEXT_PREFIX, bom_cell_text, free_text_cell
from ai_eda.compilers.spice import SpiceNetlistCompiler
from ai_eda.compilers.gerber import DrillExporter, GerberExporter
from ai_eda.compilers.project import ProjectFileCompiler

__all__ = [
    "Compiler",
    "CompileContext",
    "SchematicCompiler",
    "PCBCompiler",
    "BOMCompiler",
    "CPLCompiler",
    "TEXT_PREFIX",
    "bom_cell_text",
    "free_text_cell",
    "SpiceNetlistCompiler",
    "GerberExporter",
    "DrillExporter",
    "ProjectFileCompiler",
]
