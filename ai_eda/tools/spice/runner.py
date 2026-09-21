from __future__ import annotations

import hashlib
import shutil
import subprocess
from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from ai_eda.errors import ToolExecutionError, ToolUnavailableError


class SpiceAnalysis(StrEnum):
    OP = "op"
    DC = "dc"
    AC = "ac"
    TRAN = "tran"


class SpiceResult(BaseModel):
    engine: str
    engine_version: str
    netlist_path: str
    netlist_hash: str
    analysis: SpiceAnalysis
    #: node/branch name -> value (op) or -> list of samples (tran/ac/dc)
    data: dict[str, float | list[float]] = Field(default_factory=dict)
    raw_output_path: str | None = None
    stdout: str = ""
    succeeded: bool


class SpiceRunner(ABC):
    engine: str = "abstract"

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def version(self) -> str: ...

    @abstractmethod
    def run(self, netlist_path: Path, analysis: SpiceAnalysis, workdir: Path) -> SpiceResult: ...

    @staticmethod
    def netlist_hash(netlist_path: Path) -> str:
        return "sha256:" + hashlib.sha256(netlist_path.read_bytes()).hexdigest()


class NgspiceRunner(SpiceRunner):
    """ngspice batch-mode runner (scaffold: locates the binary, runs it, does not yet parse output)."""

    engine = "ngspice"

    def __init__(self, binary: str | None = None) -> None:
        self.binary = binary or shutil.which("ngspice")

    def available(self) -> bool:
        return self.binary is not None

    def version(self) -> str:
        if not self.available():
            raise ToolUnavailableError("ngspice not found on PATH")
        out = subprocess.run(
            [self.binary, "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30
        )
        return out.stdout.strip().splitlines()[0] if out.stdout else "unknown"

    def run(self, netlist_path: Path, analysis: SpiceAnalysis, workdir: Path) -> SpiceResult:
        if not self.available():
            raise ToolUnavailableError("ngspice not found on PATH")
        workdir.mkdir(parents=True, exist_ok=True)
        raw = workdir / (netlist_path.stem + ".raw")
        proc = subprocess.run(
            [self.binary, "-b", "-r", str(raw), str(netlist_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=workdir,
            timeout=600,
        )
        if proc.returncode != 0:
            raise ToolExecutionError(f"ngspice exited {proc.returncode}: {proc.stderr[-2000:]}")
        # TODO: parse the rawfile into ``data`` (ASCII/binary rawfile reader).
        return SpiceResult(
            engine=self.engine,
            engine_version=self.version(),
            netlist_path=str(netlist_path),
            netlist_hash=self.netlist_hash(netlist_path),
            analysis=analysis,
            raw_output_path=str(raw) if raw.exists() else None,
            stdout=proc.stdout,
            succeeded=True,
        )
