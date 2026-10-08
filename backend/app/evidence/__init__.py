"""TruthLens Evidence Retrieval, Claim Extraction, and Stance Analysis Package."""

from backend.app.evidence.claim_extractor import ClaimExtractor, ClaimProposition
from backend.app.evidence.provenance import (
    SourceProvenance,
    extract_domain,
    get_source_provenance,
    is_safe_external_url,
)
from backend.app.evidence.retriever import EvidenceRetriever, RetrievedEvidence
from backend.app.evidence.stance_classifier import (
    EvidenceSufficiencyProfile,
    StanceClassifier,
    StanceResult,
)

__all__ = [
    "ClaimExtractor",
    "ClaimProposition",
    "EvidenceRetriever",
    "EvidenceSufficiencyProfile",
    "RetrievedEvidence",
    "SourceProvenance",
    "StanceClassifier",
    "StanceResult",
    "extract_domain",
    "get_source_provenance",
    "is_safe_external_url",
]
