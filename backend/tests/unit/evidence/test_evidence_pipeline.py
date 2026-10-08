"""Unit tests for Phase 6: Claim Extraction, Evidence Retrieval, Provenance, and Stance Analysis."""


from backend.app.evidence.claim_extractor import ClaimExtractor
from backend.app.evidence.provenance import (
    get_source_provenance,
    is_safe_external_url,
)
from backend.app.evidence.retriever import EvidenceRetriever
from backend.app.evidence.stance_classifier import StanceClassifier
from backend.app.models.entities import StanceType


def test_claim_extraction_and_decomposition():
    """Verify claim proposition decomposition, offset tracking, and search query formulation."""
    text = (
        "NASA confirmed the discovery of water ice on the Moon in a new official report yesterday. "
        "I personally think space exploration is very exciting. "
        "The mission budget increased by 15 percent according to congressional testimony."
    )
    claims = ClaimExtractor.extract_claims(text, max_claims=5)

    assert len(claims) >= 2
    # Find the verifiable NASA sentence
    nasa_claim = next(c for c in claims if "NASA" in c.claim_text or "Moon" in c.claim_text)
    assert nasa_claim.start_char >= 0
    assert nasa_claim.end_char > nasa_claim.start_char
    assert nasa_claim.salience_score > 0.50
    assert len(nasa_claim.search_queries) >= 1
    assert any("fact check" in q for q in nasa_claim.search_queries)

    # Subjective sentence ("I personally think...") should be filtered out
    claim_texts = [c.claim_text.lower() for c in claims]
    assert not any("i personally think" in ct for ct in claim_texts)


def test_ssrf_protection():
    """Verify SSRF mitigation blocks internal, loopback, and metadata network targets."""
    # Malicious / internal targets
    assert not is_safe_external_url("http://localhost:8000/api")
    assert not is_safe_external_url("http://127.0.0.1:5432")
    assert not is_safe_external_url("http://0.0.0.0/")
    assert not is_safe_external_url("http://169.254.169.254/latest/meta-data")
    assert not is_safe_external_url("file:///etc/passwd")
    assert not is_safe_external_url("ftp://internal.server/data")

    # Safe public domains
    assert is_safe_external_url("https://www.reuters.com/fact-check")
    assert is_safe_external_url("https://apnews.com/hub/ap-fact-check")
    assert is_safe_external_url("https://www.who.int/news")


def test_source_provenance_and_reliability():
    """Verify source domain resolution and accredited fact-checker detection."""
    prov_reuters = get_source_provenance("https://www.reuters.com/article/idUS1234")
    assert prov_reuters.domain == "reuters.com"
    assert prov_reuters.is_fact_checker is True
    assert prov_reuters.reliability_score >= 0.90

    prov_satire = get_source_provenance("https://www.theonion.com/article/fake-story")
    assert prov_satire.domain == "theonion.com"
    assert prov_satire.reliability_score <= 0.10

    prov_general = get_source_provenance("https://example-blog.org/post/1")
    assert prov_general.domain == "example-blog.org"
    assert prov_general.reliability_score == 0.50


def test_stance_classification():
    """Verify NLI stance detection distinguishes SUPPORTS, CONTRADICTS, and NEUTRAL."""
    claim = "NASA announced discovery of alien civilizations on Mars"

    refuting_passage = (
        "Reports circulating on social media claiming NASA discovered alien cities on Mars are false. "
        "NASA scientists officially confirmed no extraterrestrial structures or civilizations exist."
    )
    stance_contra = StanceClassifier.classify_stance(claim, refuting_passage)
    assert stance_contra.stance == StanceType.CONTRADICTS
    assert stance_contra.confidence >= 0.65
    assert stance_contra.contradiction_score > stance_contra.support_score

    supporting_claim = "Congress passed bipartisan infrastructure legislation"
    supporting_passage = "Official records confirmed that Congress passed the bipartisan infrastructure law with majority approval."
    stance_supp = StanceClassifier.classify_stance(supporting_claim, supporting_passage)
    assert stance_supp.stance == StanceType.SUPPORTS
    assert stance_supp.confidence >= 0.65
    assert stance_supp.support_score > stance_supp.contradiction_score

    neutral_passage = "NASA operates several planetary exploration programs throughout the solar system."
    stance_neut = StanceClassifier.classify_stance(claim, neutral_passage)
    assert stance_neut.stance == StanceType.NEUTRAL


def test_evidence_retriever_local_archive_and_ranking():
    """Verify multi-source retrieval against curated archive and relevance ranking."""
    claim_id = "test_claim_01"
    claim_text = "NASA discovered alien cities on Mars with rovers"

    evidence_items = EvidenceRetriever.retrieve_evidence_for_claim(
        claim_id=claim_id,
        claim_text=claim_text,
        max_evidence=3,
        enable_live_web=False,  # Test offline archive
    )

    assert len(evidence_items) >= 1
    top_item = evidence_items[0]
    assert top_item.claim_id == claim_id
    assert "reuters.com" in top_item.source_url
    assert top_item.relevance_score > 0.40
    assert top_item.stance == StanceType.CONTRADICTS
    assert top_item.reliability_score >= 0.90


def test_evidence_sufficiency_and_conflict_formulation():
    """Verify mathematical Conflict Index and Evidence Sufficiency formulations."""
    # Scenario 1: Unanimous contradiction -> REFUTED
    evals_refuted = [
        {
            "relevance_score": 0.90,
            "stance": StanceType.CONTRADICTS,
            "support_score": 0.05,
            "contradiction_score": 0.85,
            "reliability_score": 0.95,
        },
        {
            "relevance_score": 0.85,
            "stance": StanceType.CONTRADICTS,
            "support_score": 0.05,
            "contradiction_score": 0.80,
            "reliability_score": 0.90,
        },
    ]
    profile_refuted = StanceClassifier.evaluate_sufficiency_and_consensus(evals_refuted)
    assert profile_refuted.verdict == "REFUTED"
    assert profile_refuted.sufficiency_score >= 0.70
    assert profile_refuted.conflict_score < 0.30

    # Scenario 2: Conflicting evidence -> MIXED
    evals_mixed = [
        {
            "relevance_score": 0.85,
            "stance": StanceType.SUPPORTS,
            "support_score": 0.80,
            "contradiction_score": 0.10,
            "reliability_score": 0.90,
        },
        {
            "relevance_score": 0.85,
            "stance": StanceType.CONTRADICTS,
            "support_score": 0.10,
            "contradiction_score": 0.80,
            "reliability_score": 0.90,
        },
    ]
    profile_mixed = StanceClassifier.evaluate_sufficiency_and_consensus(evals_mixed)
    assert profile_mixed.verdict == "MIXED"
    assert profile_mixed.conflict_score >= 0.40

    # Scenario 3: Zero or weak evidence -> INSUFFICIENT_EVIDENCE
    evals_insufficient = [
        {
            "relevance_score": 0.20,
            "stance": StanceType.NEUTRAL,
            "support_score": 0.10,
            "contradiction_score": 0.10,
            "reliability_score": 0.50,
        }
    ]
    profile_insufficient = StanceClassifier.evaluate_sufficiency_and_consensus(evals_insufficient)
    assert profile_insufficient.verdict == "INSUFFICIENT_EVIDENCE"
    assert profile_insufficient.sufficiency_score < 0.40
