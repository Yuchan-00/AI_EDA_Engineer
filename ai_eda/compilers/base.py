from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field

from ai_eda.ir import ArtifactKind, ArtifactRef, CircuitIR


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

    def _write(self, ir: CircuitIR, path: Path, content: str | bytes) -> ArtifactRef:
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
        )
