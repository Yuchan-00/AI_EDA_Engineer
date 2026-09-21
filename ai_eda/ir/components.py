"""Component model.

Rule: an MPN, package, pinout, price or stock figure that could not be
confirmed against a datasheet / official distributor data is stored with
``llm_generated`` or ``assumption`` provenance (or left ``None``). The BOM
compiler and the reviewer refuse to treat such values as facts.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from ai_eda.ir.provenance import Provenance, ProvenanceKind, SourceRef, Traced


class PinElectricalType(StrEnum):
    """Mirrors KiCad pin electrical types so ERC semantics carry through."""

    INPUT = "input"
    OUTPUT = "output"
    BIDIRECTIONAL = "bidirectional"
    TRI_STATE = "tri_state"
    PASSIVE = "passive"
    FREE = "free"
    UNSPECIFIED = "unspecified"
    POWER_IN = "power_in"
    POWER_OUT = "power_out"
    OPEN_COLLECTOR = "open_collector"
    OPEN_EMITTER = "open_emitter"
    NO_CONNECT = "no_connect"


class Pin(BaseModel):
    number: str  # "1", "A3", ...
    name: str
    electrical_type: PinElectricalType = PinElectricalType.PASSIVE
    #: provenance of the pin definition (datasheet page, KiCad symbol, or a model guess)
    provenance: Provenance


class LibraryRef(BaseModel):
    """Reference to a KiCad symbol or footprint.

    ``verified`` is only True after :mod:`ai_eda.tools.kicad.library` has
    actually found the entry in an installed library.
    """

    library: str  # e.g. "Device", "Package_TO_SOT_SMD"
    name: str  # e.g. "R", "SOT-23"
    verified: bool = False
    library_path: str | None = None


class SourcingInfo(BaseModel):
    supplier: str  # "JLCPCB", "LCSC", "Digi-Key", ...
    supplier_part_number: Traced[str] | None = None
    stock: Traced[int] | None = None
    unit_price: Traced[float] | None = None
    currency: str | None = None
    #: e.g. JLCPCB "Basic" / "Extended"
    assembly_class: Traced[str] | None = None


class Component(BaseModel):
    ref: str  # reference designator: R1, U3, ...
    value: str  # "10k", "LM7805", ...
    description: str = ""
    manufacturer: Traced[str] | None = None
    mpn: Traced[str] | None = None
    datasheet: SourceRef | None = None
    package: Traced[str] | None = None
    pins: list[Pin] = Field(default_factory=list)
    #: key -> traced characteristic, e.g. "v_max", "i_max", "power_rating", "tolerance"
    electrical: dict[str, Traced] = Field(default_factory=dict)
    symbol: LibraryRef | None = None
    footprint: LibraryRef | None = None
    sourcing: list[SourcingInfo] = Field(default_factory=list)
    #: why this part is in the design (which requirement / block it serves)
    provenance: Provenance
    #: requirement ids this component satisfies - used for Requirements <-> IR review
    serves_requirements: list[str] = Field(default_factory=list)

    def pin(self, number: str) -> Pin | None:
        for p in self.pins:
            if p.number == number:
                return p
        return None

    @property
    def has_authoritative_identity(self) -> bool:
        """True when MPN is backed by a datasheet or official part data."""
        return bool(self.mpn and self.mpn.provenance.kind == ProvenanceKind.AUTHORITATIVE)
