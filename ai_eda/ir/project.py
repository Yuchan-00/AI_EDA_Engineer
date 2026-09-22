"""Top-level IR container and derived-artifact bookkeeping."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from ai_eda.ir.components import Component
from ai_eda.ir.constraints import Constraint
from ai_eda.ir.nets import Net, PinRef
from ai_eda.ir.pcb import PCBDesign
from ai_eda.ir.provenance import Traced
from ai_eda.ir.regulatory import RegulatoryState
from ai_eda.ir.requirements import RequirementSet
from ai_eda.ir.simulation import SimulationSetup
from ai_eda.ir.topology import Topology
from ai_eda.ir.validation import ValidationState

IR_SCHEMA_VERSION = "0.1"

#: project bookkeeping that says where / when the IR was built, not what it designs (not hashed)
NON_DESIGN_PROJECT_FIELDS: tuple[str, ...] = ("workdir", "created_at")
#: requirement bookkeeping that memoises what a model *said* (``RequirementSet.extraction_cache``), not what
#: the user asked for or what entered ``requirements`` (not hashed; ``corrections`` are the user's words and stay)
NON_DESIGN_REQUIREMENT_FIELDS: tuple[str, ...] = ("extraction_cache",)
#: wall-clock keys stripped from every nested object before hashing (``Provenance.created_at`` defaults to
#: *now*, so two identical designs built a millisecond apart would otherwise never share a hash)
WALL_CLOCK_KEYS: frozenset[str] = frozenset({"created_at"})


def strip_wall_clock(obj):
    """``obj`` (JSON-able data) without any :data:`WALL_CLOCK_KEYS` entry at any depth."""
    if isinstance(obj, dict):
        return {k: strip_wall_clock(v) for k, v in obj.items() if k not in WALL_CLOCK_KEYS}
    if isinstance(obj, list):
        return [strip_wall_clock(v) for v in obj]
    return obj


class ArtifactKind(StrEnum):
    SCHEMATIC = "kicad_sch"
    PCB = "kicad_pcb"
    SPICE_NETLIST = "spice_netlist"
    SPICE_RESULT = "spice_result"
    BOM = "bom"
    CPL = "cpl"
    GERBER = "gerber"
    DRILL = "drill"
    GERBER_JOB = "gerber_job"
    ERC_REPORT = "erc_report"
    DRC_REPORT = "drc_report"
    REVIEW_REPORT = "review_report"


def hash_file_set(paths: Iterable[Path]) -> str:
    """Order-independent hash of a set of files (name + bytes of each, sorted by name).

    Used for multi-file artifacts such as a gerber set, where no single file
    represents the whole. The directory location does not enter the hash.
    """
    h = hashlib.sha256()
    for p in sorted((Path(p) for p in paths), key=lambda p: p.name):
        h.update(p.name.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return "sha256:" + h.hexdigest()


class ArtifactRef(BaseModel):
    """A derived file plus the IR hash it was generated from.

    If ``generated_from_ir_hash != ir.content_hash()`` the artifact is stale
    and the reviewer flags it; the repair strategy is simply "regenerate".

    A multi-file artifact (a gerber set) lists its member files in ``files``;
    ``path`` then points at the file a human opens first (the ``.gbrjob``
    manifest) and ``content_hash`` covers the whole set via
    :func:`hash_file_set`.
    """

    kind: ArtifactKind
    path: str
    content_hash: str | None = None
    generated_from_ir_hash: str | None = None
    generator: str | None = None  # compiler / tool id
    generator_version: str | None = None
    #: member files of a multi-file artifact (absolute paths); empty for a single file
    files: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def is_stale(self, ir_hash: str) -> bool:
        return self.generated_from_ir_hash != ir_hash

    def disk_hash(self) -> str | None:
        """Hash of what is on disk right now (``None`` if any member file is missing)."""
        if self.files:
            paths = [Path(f) for f in self.files]
            if not all(p.is_file() for p in paths):
                return None
            return hash_file_set(paths)
        p = Path(self.path)
        if not p.is_file():
            return None
        return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()

    def matches_disk(self) -> bool:
        if self.content_hash is None:
            return False
        return self.disk_hash() == self.content_hash


class ProjectMeta(BaseModel):
    id: str
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    workdir: str | None = None


class CircuitIR(BaseModel):
    schema_version: str = IR_SCHEMA_VERSION
    project: ProjectMeta
    requirements: RequirementSet = Field(default_factory=RequirementSet)
    topology: Topology | None = None
    components: list[Component] = Field(default_factory=list)
    nets: list[Net] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    #: design parameters: "v_in", "i_out", "r1", "p_diss" ... all traced
    parameters: dict[str, Traced] = Field(default_factory=dict)
    pcb: PCBDesign | None = None
    #: stimuli / analyses / expectations for the SPICE stage (part of the design hash)
    simulation: SimulationSetup | None = None
    regulatory: RegulatoryState = Field(default_factory=RegulatoryState)
    validation: ValidationState = Field(default_factory=ValidationState)
    artifacts: dict[ArtifactKind, ArtifactRef] = Field(default_factory=dict)

    # --- lookups -------------------------------------------------------------

    def component(self, ref: str) -> Component | None:
        for c in self.components:
            if c.ref == ref:
                return c
        return None

    def net(self, name: str) -> Net | None:
        for n in self.nets:
            if n.name == name:
                return n
        return None

    def net_of(self, pin: PinRef) -> Net | None:
        for n in self.nets:
            for p in n.pins:
                if p.component_ref == pin.component_ref and p.pin_number == pin.pin_number:
                    return n
        return None

    # --- hashing -------------------------------------------------------------

    #: fields that describe *state about* the design rather than the design itself
    _NON_DESIGN_FIELDS = ("validation", "artifacts")

    def design_dict(self) -> dict:
        """The design content as JSON-able data: what :meth:`content_hash` hashes.

        Excludes ``validation`` / ``artifacts`` (state about the design),
        ``project.workdir`` / ``project.created_at`` (where and when it was
        built), ``requirements.extraction_cache`` (what a model said, not
        what the design is) and every ``created_at`` timestamp, so the same
        design built twice - in another folder, at another time, or loaded
        from disk - hashes the same. ``SourceRef.retrieved_at`` stays: which
        retrieval of a datasheet was used is design provenance, and it is
        never filled in by a clock default.
        """
        data = self.model_dump(mode="json", exclude=set(self._NON_DESIGN_FIELDS))
        for key in NON_DESIGN_PROJECT_FIELDS:
            data["project"].pop(key, None)
        for key in NON_DESIGN_REQUIREMENT_FIELDS:
            data["requirements"].pop(key, None)
        return strip_wall_clock(data)

    def content_hash(self) -> str:
        """Stable hash of the design content (see :meth:`design_dict` for what is excluded)."""
        payload = json.dumps(self.design_dict(), sort_keys=True, separators=(",", ":"), default=str)
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # --- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> "CircuitIR":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
