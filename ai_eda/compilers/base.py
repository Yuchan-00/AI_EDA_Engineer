from __future__ import annotations

import math

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from ai_eda.errors import CompileError
from pydantic import BaseModel, Field

from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR


def check_finite(data, where: str) -> None:
    """:class:`~ai_eda.errors.CompileError` when any float in ``data`` (JSON-able) is NaN / infinite, naming its path.

    A NaN placement would write ``(at nan 6)`` - a file KiCad cannot load - or
    ``nanmm`` into a CPL sent to a fab; an infinite angle crashes the angle
    normalisation. Nothing is guessed: the item is named and refused.
    """
    if isinstance(data, dict):
        for k, v in data.items():
            check_finite(v, f"{where}.{k}")
    elif isinstance(data, (list, tuple)):
        for i, v in enumerate(data):
            check_finite(v, f"{where}[{i}]")
    elif isinstance(data, float) and not math.isfinite(data):
        raise CompileError(f"{where} is {data!r}: not a finite number")


class CompileContext(BaseModel):
    workdir: Path
    #: injected helpers: {"kicad_library": KicadLibrary, ...}
    tools: dict[str, object] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class Compiler(ABC):
    id: str = "compiler.base"
    version: str = "0.1"
    kind: ArtifactKind

    @abstractmethod
    def compile(self, ir: CircuitIR, ctx: CompileContext) -> ArtifactRef:
        """Generate the artifact and return a reference stamped with the IR hash."""

    # helper --------------------------------------------------------------------

    def _write(self, ir: CircuitIR, path: Path, content: str | bytes, notes: Iterable[str] = ()) -> ArtifactRef:
        """Write ``content`` and return its stamped reference; ``notes``: what the compiler altered while writing (a human reads them in ``compile.<kind>``)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8") if isinstance(content, str) else content
        path.write_bytes(data)
        return ArtifactRef(
            kind=self.kind,
            path=str(path),
            content_hash="sha256:" + hashlib.sha256(data).hexdigest(),
            generated_from_ir_hash=ir.content_hash(),
            generator=self.id,
            generator_version=self.version,
            notes=list(notes),
        )
