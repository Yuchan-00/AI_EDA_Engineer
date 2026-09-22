"""Universal Circuit IR.

The IR is the *only* original design data. ``.kicad_sch``, ``.kicad_pcb``,
BOM, CPL, SPICE netlists, Gerbers and drill files are all derived from it and
can be regenerated from it at any time (this is what makes deterministic
repair possible - see :mod:`ai_eda.repair`).
"""

from ai_eda.ir.provenance import (
    ProvenanceKind,
    SourceRef,
    Provenance,
    Traced,
    authoritative,
    assumption,
    derived,
    llm_generated,
    user_requirement,
)
from ai_eda.ir.validation import (
    ValidationStatus,
    Evidence,
    ValidationResult,
    ValidationState,
    worst_status,
)
from ai_eda.ir.requirements import (
    RequirementKind,
    RequirementStatus,
    Requirement,
    MissingInformation,
    RequirementConflict,
    RequirementSet,
)
from ai_eda.ir.components import (
    PinElectricalType,
    Pin,
    LibraryRef,
    SourcingInfo,
    Component,
)
from ai_eda.ir.nets import NetKind, PinRef, Net
from ai_eda.ir.simulation import (
    SpiceDevice,
    SpiceBinding,
    StimulusKind,
    Stimulus,
    AnalysisSpec,
    Reduce,
    Expectation,
    SimulationSetup,
)
from ai_eda.ir.topology import CircuitDomain, Block, Topology
from ai_eda.ir.constraints import ConstraintKind, Constraint
from ai_eda.ir.pcb import (
    UNRECORDED_ORIGIN,
    BoardSide,
    Layer,
    BoardOutline,
    Placement,
    Track,
    Via,
    Zone,
    ManufacturingConstraints,
    PCBDesign,
    unrecorded_origin,
)
from ai_eda.ir.regulatory import (
    Jurisdiction,
    RegulatoryProvenance,
    RegulatoryRequirement,
    RegulatoryState,
)
from ai_eda.ir.project import ArtifactKind, ArtifactRef, ProjectMeta, CircuitIR, hash_file_set

__all__ = [
    "ProvenanceKind", "SourceRef", "Provenance", "Traced",
    "authoritative", "assumption", "derived", "llm_generated", "user_requirement",
    "ValidationStatus", "Evidence", "ValidationResult", "ValidationState", "worst_status",
    "RequirementKind", "RequirementStatus", "Requirement", "MissingInformation",
    "RequirementConflict", "RequirementSet",
    "PinElectricalType", "Pin", "LibraryRef", "SourcingInfo", "Component",
    "NetKind", "PinRef", "Net",
    "SpiceDevice", "SpiceBinding", "StimulusKind", "Stimulus", "AnalysisSpec", "Reduce",
    "Expectation", "SimulationSetup",
    "CircuitDomain", "Block", "Topology",
    "ConstraintKind", "Constraint",
    "BoardSide", "Layer", "BoardOutline", "Placement", "Track", "Via", "Zone",
    "ManufacturingConstraints", "PCBDesign", "UNRECORDED_ORIGIN", "unrecorded_origin",
    "Jurisdiction", "RegulatoryProvenance", "RegulatoryRequirement", "RegulatoryState",
    "ArtifactKind", "ArtifactRef", "ProjectMeta", "CircuitIR", "hash_file_set",
]
