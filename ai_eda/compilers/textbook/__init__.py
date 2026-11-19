"""Textbook schematic layout, version 1.0: every schematic drawn block by block in textbook style.

:func:`build` is the entry point (``SchematicCompiler`` -> :class:`DrawResult`). The stages are in
:mod:`~ai_eda.compilers.textbook.pipeline`, the data model in :mod:`~ai_eda.compilers.textbook.model`. A layout reason
(:class:`LayoutFailure`) never leaves :func:`build`: it sends the drawing down the fallback ladder, whose last rung is the
legacy table inside a 1.0 artifact, and the notes say which rungs were taken.
"""

from ai_eda.compilers.textbook.model import DrawResult, LayoutFailure
from ai_eda.compilers.textbook.pipeline import VERSION, build

__all__ = ["VERSION", "build", "LayoutFailure", "DrawResult"]
