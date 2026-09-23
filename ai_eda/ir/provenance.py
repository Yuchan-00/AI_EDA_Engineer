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
import math
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field, SerializationInfo, SerializerFunctionWrapHandler, field_validator, model_serializer

#: ``model_dump(context={"view": DESIGN_VIEW})`` renders a model as *design content*: the models below drop their
#: wall-clock, locator and verification-outcome fields in that view, and :meth:`ai_eda.ir.CircuitIR.design_dict`
#: hashes exactly that view. A field is excluded because of what it is, never because of what a key is called.
DESIGN_VIEW = "design"


def in_design_view(info: SerializationInfo) -> bool:
    ctx = info.context
    return isinstance(ctx, dict) and ctx.get("view") == DESIGN_VIEW


def design_data(model: BaseModel) -> dict:
    """``model`` as JSON-able design content (the design view)."""
    return model.model_dump(mode="json", context={"view": DESIGN_VIEW})


def drop_in_design_view(*names: str):
    """A ``model_serializer`` that leaves ``names`` out of the design view (they are not design content)."""

    def _serialize(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo):
        data = handler(self)
        if in_design_view(info):
            for name in names:
                data.pop(name, None)
        return data

    return model_serializer(mode="wrap")(_serialize)


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
    #: local archived copy - a locator, not design content: the same design archived under another workdir hashes
    #: the same; ``content_hash`` (which stays) pins the document
    document_path: str | None = None
    content_hash: str | None = None  # sha256 of the archived document
    retrieved_at: datetime | None = None  # when it was fetched: a clock, not the design (out of the design view)

    _design = drop_in_design_view("document_path", "retrieved_at")

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        return "sha256:" + hashlib.sha256(data).hexdigest()

    @staticmethod
    def from_document(doc: Any, section: str | None = None, title: str | None = None, authority: str | None = None) -> "SourceRef":
        """A reference to an archived document (:class:`ai_eda.tools.sources.ArchivedDocument`): its final URL, path, hash and retrieval time.

        ``section`` names where in the document the claim stands (``"page 3"``,
        an article heading). The document object is duck-typed (it must offer
        ``source_ref(title, section, authority)``) so the IR does not import the
        tools layer.
        """
        return doc.source_ref(title=title, section=section, authority=authority)


class Provenance(BaseModel):
    kind: ProvenanceKind
    source: SourceRef | None = None
    #: ids / keys of the traced values this one was derived from (in the calculator's parameter order)
    derived_from: list[str] = Field(default_factory=list)
    #: calculator role -> id, written by the calculator itself (``{"v_in": "v_in", "r1": "r1", "r2": "r2"}``);
    #: :func:`ai_eda.tools.calc.recompute_parameters` rebuilds the call from these roles instead of trusting
    #: the positional order of ``derived_from``. Empty for a value nobody recorded roles for.
    inputs: dict[str, str] = Field(default_factory=dict)
    #: deterministic tool that produced a derived value (e.g. "calc.voltage_divider", "ngspice")
    tool: str | None = None
    tool_version: str | None = None
    #: free-form rationale; for assumptions this is what the user must confirm
    note: str | None = None
    #: wall-clock bookkeeping; excluded from :meth:`ai_eda.ir.CircuitIR.content_hash` (it is not design content)
    created_at: datetime = Field(default_factory=_now)

    _design = drop_in_design_view("created_at")

    @property
    def is_authoritative(self) -> bool:
        return self.kind in (ProvenanceKind.USER_REQUIREMENT, ProvenanceKind.AUTHORITATIVE)

    @property
    def needs_verification(self) -> bool:
        return self.kind in (ProvenanceKind.LLM_GENERATED, ProvenanceKind.ASSUMPTION)


T = TypeVar("T")


def _non_finite(v: Any) -> bool:
    """Whether ``v`` is, or contains, a float that is not a number JSON can carry (``inf`` / ``nan``)."""
    if isinstance(v, float):
        return not math.isfinite(v)
    if isinstance(v, (list, tuple)):
        return any(_non_finite(x) for x in v)
    if isinstance(v, dict):
        return any(_non_finite(x) for x in v.values())
    return False


def _jsonish(v: Any) -> Any:
    """``v`` with tuples turned into lists at any depth (what JSON would make of them)."""
    if isinstance(v, (list, tuple)):
        return [_jsonish(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonish(x) for k, x in v.items()}
    return v


class Traced(BaseModel, Generic[T]):
    """A value plus its unit and provenance.

    Anything that matters to the design (a voltage, an MPN, a pin count, a
    regulatory limit) is stored as ``Traced`` rather than as a bare value.
    """

    value: T
    unit: str | None = None
    provenance: Provenance

    @field_validator("value", mode="before")
    @classmethod
    def _honest_value(cls, v: Any) -> Any:
        """No silent coercion into a numeric fact, and JSON-shaped containers.

        ``Traced[float]`` / ``Traced[int]`` refuse a ``str`` (``"5"``) and a
        ``bool`` (``True`` would read as 1.0): a number that arrives as text
        from a model or a hand edit is not a measured value with this
        provenance. Tuples become lists so a value compares and hashes the
        same before and after a JSON round trip. ``inf`` / ``nan`` are refused
        at any depth: JSON has no token for them, so ``save`` would write
        ``null`` and the project would not load again.
        """
        annotation = cls.model_fields["value"].annotation
        if annotation in (float, int) and isinstance(v, (bool, str)):
            raise ValueError(f"{annotation.__name__} value must be a number, not {type(v).__name__} {v!r}")
        if _non_finite(v):
            raise ValueError(f"a traced number must be finite; got {v!r}")
        return _jsonish(v)

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
    derived_from: list[str] | None = None,
    unit: str | None = None,
    tool_version: str | None = None,
    note: str | None = None,
    inputs: dict[str, str] | None = None,
) -> Traced[T]:
    """A value computed by ``tool``.

    ``inputs`` maps the tool's input roles to the ids that filled them
    (``{"v_in": "v_in", "r1": "r1", "r2": "r2"}``); ``derived_from`` then
    defaults to those ids in role order. Giving both is allowed only when
    they agree (``ValueError`` otherwise), so a provenance can never name one
    input list and another role mapping.
    """
    inputs = dict(inputs or {})
    if derived_from is None:
        derived_from = list(inputs.values())
    elif inputs and list(inputs.values()) != list(derived_from):
        raise ValueError(f"derived_from {list(derived_from)} and inputs {inputs} name different ids")
    return Traced(
        value=value,
        unit=unit,
        provenance=Provenance(
            kind=ProvenanceKind.DERIVED,
            tool=tool,
            tool_version=tool_version,
            derived_from=list(derived_from),
            inputs=inputs,
            note=note,
        ),
    )


def llm_generated(value: T, model: str, unit: str | None = None, note: str | None = None) -> Traced[T]:
    return Traced(
        value=value,
        unit=unit,
        provenance=Provenance(kind=ProvenanceKind.LLM_GENERATED, tool=model, note=note),
    )
