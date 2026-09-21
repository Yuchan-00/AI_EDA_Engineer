"""Fab capability model.

A capability value only becomes a design rule once it is ``authoritative``
(fetched from the vendor's official capability page with URL/hash/date).
Until then the whole capability set is NOT_VERIFIED and the manufacturing
stage must say so instead of declaring the board manufacturable.
"""

from __future__ import annotations

from pydantic import BaseModel

from ai_eda.ir import ManufacturingConstraints, ProvenanceKind, SourceRef, ValidationStatus


class FabCapability(BaseModel):
    fab: str
    constraints: ManufacturingConstraints
    source: SourceRef | None = None

    def verification_status(self) -> ValidationStatus:
        traced = [
            v for v in self.constraints.model_dump(exclude={"fab"}).values() if v is not None
        ]
        if not traced:
            return ValidationStatus.NOT_VERIFIED
        kinds = {
            getattr(getattr(self.constraints, k), "provenance").kind
            for k, v in self.constraints.model_dump(exclude={"fab"}).items()
            if v is not None
        }
        return ValidationStatus.PASS if kinds == {ProvenanceKind.AUTHORITATIVE} else ValidationStatus.NOT_VERIFIED


def jlcpcb_capability_unverified() -> FabCapability:
    """Placeholder: no limits filled in. They must be fetched from JLCPCB's official capability page."""
    return FabCapability(fab="JLCPCB", constraints=ManufacturingConstraints(fab="JLCPCB"))
