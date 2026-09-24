"""Top-level IR container and derived-artifact bookkeeping."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, Field

from ai_eda.errors import IRSchemaError
from ai_eda.ir.components import Component
from ai_eda.ir.constraints import Constraint
from ai_eda.ir.nets import Net, PinRef
from ai_eda.ir.pcb import PCBDesign
from ai_eda.ir.provenance import DESIGN_VIEW, Traced, design_data, drop_in_design_view
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


def unknown_keys(raw, dumped, path: str = "") -> list[str]:
    """Dotted paths of keys present in ``raw`` (loaded JSON) that ``dumped`` (the validated model dumped back) lacks."""
    out: list[str] = []
    if isinstance(raw, dict) and isinstance(dumped, dict):
        for k, v in raw.items():
            here = f"{path}.{k}" if path else str(k)
            if k not in dumped:
                out.append(here)
            else:
                out.extend(unknown_keys(v, dumped[k], here))
    elif isinstance(raw, list) and isinstance(dumped, list):
        for i, (a, b) in enumerate(zip(raw, dumped)):
            out.extend(unknown_keys(a, b, f"{path}[{i}]"))
    return out


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

    ``notes`` lists what the generator altered or left out while writing the
    file (a BOM free-text cell written with a leading apostrophe); it is
    bookkeeping outside the design view, never design content. Files without
    the key load (it defaults); a file that carries it is refused by code
    older than the key (``load`` drops no unknown key), by design.
    """

    kind: ArtifactKind
    path: str
    content_hash: str | None = None
    generated_from_ir_hash: str | None = None
    generator: str | None = None  # compiler / tool id
    generator_version: str | None = None
    #: member files of a multi-file artifact (absolute paths); empty for a single file
    files: list[str] = Field(default_factory=list)
    #: what the generator altered or left out while writing the file, for a human (e.g. BOM cells neutralised); bookkeeping, never design content
    notes: list[str] = Field(default_factory=list)
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

    _design = drop_in_design_view(*NON_DESIGN_PROJECT_FIELDS)


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
        """The design content as JSON-able data: what :meth:`content_hash` hashes (the *design view*).

        Left out, each by the model that owns it (:func:`~ai_eda.ir.provenance.drop_in_design_view`):
        ``validation`` / ``artifacts`` (state about the design);
        ``project.workdir`` / ``project.created_at`` (where and when it was built);
        ``requirements.extraction_cache`` (what a model said, not what the design is);
        every ``Provenance.created_at`` (a clock default);
        the locators ``SourceRef.document_path``, ``LibraryRef.library_path`` and
        ``RegulatoryProvenance.source_document`` (where a copy lives - the hashes that pin the copies stay);
        and the regulatory verification outcomes (``RegulatoryRequirement.status`` / ``source_status``,
        ``RegulatoryProvenance.verification_status``, ``GroundedQuote.found`` / ``page`` / ``context``), which an
        offline and an online run of the same design fill in differently.
        So the same design built twice - in another folder, at another time, against another KiCad install, or
        loaded from disk - hashes the same. ``SourceRef.retrieved_at`` and ``content_hash`` stay: which document
        version was used is design provenance, and neither is filled in by a clock default. A user parameter that
        happens to be called ``created_at`` is design content and is hashed.
        """
        return self.model_dump(mode="json", exclude=set(self._NON_DESIGN_FIELDS), context={"view": DESIGN_VIEW})

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
        """The IR saved at ``path``, or :class:`~ai_eda.errors.IRSchemaError` when it is not one this code can read faithfully.

        Pydantic ignores keys a model does not declare, so a misspelled key in
        a hand-edited file (``serves_requirement``) would silently delete
        design or traceability data and the truncated IR would then be treated
        as the design; likewise a file written by another schema version. Both
        are refused with the offending paths named.
        """
        def constant(token: str):  # Python's json accepts the non-standard Infinity / NaN tokens; the IR holds no such number
            raise IRSchemaError(f"{path}: JSON token {token} is not a number the IR can hold")

        raw = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=constant)
        if not isinstance(raw, dict):
            raise IRSchemaError(f"{path}: not an IR object")
        version = raw.get("schema_version")
        if version != IR_SCHEMA_VERSION:
            raise IRSchemaError(f"{path}: schema_version {version!r} is not {IR_SCHEMA_VERSION!r} (this code reads no other version)")
        ir = cls.model_validate(raw)
        unknown = unknown_keys(raw, ir.model_dump(mode="json"))
        if unknown:
            raise IRSchemaError(f"{path}: unknown key(s) the IR models would drop: {', '.join(unknown)}")
        return ir
