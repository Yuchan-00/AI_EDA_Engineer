"""Provenance model - every important fact records where it came from.

The five kinds map directly to the project spec:

``user_requirement``  the user said so
``authoritative``     datasheet / official part data / official regulation / vendor data
``assumption``        an engineering assumption that must be surfaced to the user
``derived``           computed by a deterministic tool from other traced values
``llm_generated``     proposed by a model; NEVER authoritative until verified
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, Field


class ProvenanceKind(StrEnum):
    USER_REQUIREMENT = "user_requirement"
    AUTHORITATIVE = "authoritative"
    ASSUMPTION = "assumption"
    DERIVED = "derived"
    LLM_GENERATED = "llm_generated"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SourceRef(BaseModel):
    """A pointer to an external document or dataset.

    Used for datasheets, official part data, regulatory documents, vendor
    capability pages, etc. ``content_hash`` lets a reviewer prove the document
    used at design time is the one on disk now.
    """

    title: str
    url: str | None = None
    authority: str | None = None  # manufacturer, regulator, distributor, ...
    section: str | None = None
    document_path: str | None = None  # local archived copy
    content_hash: str | None = None  # sha256 of the archived document
    retrieved_at: datetime | None = None

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        return "sha256:" + hashlib.sha256(data).hexdigest()


class Provenance(BaseModel):
    kind: ProvenanceKind
    source: SourceRef | None = None
    #: ids / keys of the traced values this one was derived from
    derived_from: list[str] = Field(default_factory=list)
    #: deterministic tool that produced a derived value (e.g. "calc.voltage_divider", "ngspice")
    tool: str | None = None
    tool_version: str | None = None
    #: free-form rationale; for assumptions this is what the user must confirm
    note: str | None = None
    created_at: datetime = Field(default_factory=_now)

    @property
    def is_authoritative(self) -> bool:
        return self.kind in (ProvenanceKind.USER_REQUIREMENT, ProvenanceKind.AUTHORITATIVE)

    @property
    def needs_verification(self) -> bool:
        return self.kind in (ProvenanceKind.LLM_GENERATED, ProvenanceKind.ASSUMPTION)


T = TypeVar("T")


class Traced(BaseModel, Generic[T]):
    """A value plus its unit and provenance.

    Anything that matters to the design (a voltage, an MPN, a pin count, a
    regulatory limit) is stored as ``Traced`` rather than as a bare value.
    """

    value: T
    unit: str | None = None
    provenance: Provenance

    def __str__(self) -> str:  # pragma: no cover - display only
        unit = f" {self.unit}" if self.unit else ""
        return f"{self.value}{unit} [{self.provenance.kind}]"


# --- convenience constructors -------------------------------------------------


def user_requirement(value: T, unit: str | None = None, note: str | None = None) -> Traced[T]:
    return Traced(value=value, unit=unit, provenance=Provenance(kind=ProvenanceKind.USER_REQUIREMENT, note=note))


def authoritative(value: T, source: SourceRef, unit: str | None = None, note: str | None = None) -> Traced[T]:
    return Traced(
        value=value,
        unit=unit,
        provenance=Provenance(kind=ProvenanceKind.AUTHORITATIVE, source=source, note=note),
    )


def assumption(value: T, note: str, unit: str | None = None) -> Traced[T]:
    return Traced(value=value, unit=unit, provenance=Provenance(kind=ProvenanceKind.ASSUMPTION, note=note))


def derived(
    value: T,
    tool: str,
    derived_from: list[str],
    unit: str | None = None,
    tool_version: str | None = None,
    note: str | None = None,
) -> Traced[T]:
    return Traced(
        value=value,
        unit=unit,
        provenance=Provenance(
            kind=ProvenanceKind.DERIVED,
            tool=tool,
            tool_version=tool_version,
            derived_from=derived_from,
            note=note,
        ),
    )


def llm_generated(value: T, model: str, unit: str | None = None, note: str | None = None) -> Traced[T]:
    return Traced(
        value=value,
        unit=unit,
        provenance=Provenance(kind=ProvenanceKind.LLM_GENERATED, tool=model, note=note),
    )
