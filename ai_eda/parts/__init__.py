"""Parts track: does a component exist, and what does its datasheet say?

The LLM never decides part truth here. A part's identity becomes
authoritative only when deterministic tools confirm it:

- :mod:`ai_eda.parts.pointers` - where the datasheet is expected (user URL,
  the IR's reference, the KiCad symbol's ``Datasheet`` property), never a
  model's URL.
- :mod:`ai_eda.parts.existence` - ``component.existence.<ref>``: symbol and
  footprint parsed from the KiCad libraries, the datasheet archived and
  hash-verified through :class:`~ai_eda.tools.sources.DocumentArchive`, the
  MPN found verbatim in it, the catalog row found; worst-of, every sub-check
  in the details, evidence attached.
- :mod:`ai_eda.parts.datasheet_facts` - claimed characteristics (from a model
  or a user file) accepted only when the archived text says them verbatim
  and the quantity parser re-reads the same number; accepted facts are
  ``authoritative`` Traced values with the page in their SourceRef.
- :mod:`ai_eda.parts.catalog` - a user-supplied distributor CSV export that
  backs ``SourcingInfo`` (hashed file as the source), never identity.
"""

from ai_eda.parts.catalog import CatalogError, CatalogRow, CatalogSource, mpn_key
from ai_eda.parts.datasheet_facts import (
    AcceptedFact,
    DatasheetFact,
    DatasheetFacts,
    GroundedFacts,
    apply_facts,
    facts_json_schema,
    facts_result,
    ground_facts,
    llm_fact_proposals,
    load_facts_file,
    searchable_document,
)
from ai_eda.parts.existence import ExistenceReport, SubCheck, check_component_existence, examine_component, find_mpn
from ai_eda.parts.pointers import DatasheetPointer, PointerSearch, datasheet_pointer, locate_datasheet

__all__ = [
    "AcceptedFact",
    "CatalogError",
    "CatalogRow",
    "CatalogSource",
    "DatasheetFact",
    "DatasheetFacts",
    "DatasheetPointer",
    "ExistenceReport",
    "GroundedFacts",
    "PointerSearch",
    "SubCheck",
    "apply_facts",
    "check_component_existence",
    "datasheet_pointer",
    "examine_component",
    "facts_json_schema",
    "facts_result",
    "find_mpn",
    "ground_facts",
    "llm_fact_proposals",
    "load_facts_file",
    "locate_datasheet",
    "mpn_key",
    "searchable_document",
]
