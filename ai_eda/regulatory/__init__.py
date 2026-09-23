"""Regulatory research: curated candidates, deterministic applicability, official-text grounding.

The stage never decides compliance. It (1) proposes candidate regulations
from a curated list (:mod:`~ai_eda.regulatory.candidates`), (2) decides
applicability from the user's scope answers and the IR's requirements with
declarative rules (:mod:`~ai_eda.regulatory.applicability`), and (3) fetches
each candidate's official text through the document archive and grounds the
list's claimed quotes in it (:mod:`~ai_eda.regulatory.research`). The result
is provenance and applicability; ``regulatory.compliance`` is always
``NOT_VERIFIED`` because assessing compliance needs an engineer or a
notified body.
"""

from ai_eda.regulatory.applicability import Evaluation, MissingInput, evaluate, yes_no
from ai_eda.regulatory.candidates import (
    DEFAULT_CANDIDATES_PATH,
    ApplicabilityRule,
    CandidateDocument,
    CandidateList,
    GroundingQuote,
    RegulatoryCandidate,
    ScopeQuestion,
    load_candidates,
)
from ai_eda.regulatory.research import (
    APPLICABILITY_CHECK,
    COMPLIANCE_CHECK,
    RESEARCH_CHECK,
    RESEARCH_TOOL,
    RESEARCH_VERSION,
    SOURCES_CHECK,
    ResearchOutcome,
    research,
)

__all__ = [
    "APPLICABILITY_CHECK",
    "COMPLIANCE_CHECK",
    "DEFAULT_CANDIDATES_PATH",
    "RESEARCH_CHECK",
    "RESEARCH_TOOL",
    "RESEARCH_VERSION",
    "SOURCES_CHECK",
    "ApplicabilityRule",
    "CandidateDocument",
    "CandidateList",
    "Evaluation",
    "GroundingQuote",
    "MissingInput",
    "RegulatoryCandidate",
    "ResearchOutcome",
    "ScopeQuestion",
    "evaluate",
    "load_candidates",
    "research",
    "yes_no",
]
