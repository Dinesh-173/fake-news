"""Unit tests for Phase 8: Evidence-Aware Multimodal Fusion and Ablation Framework."""

from backend.app.evidence.retriever import RetrievedEvidence
from backend.app.models.entities import CredibilityState, StanceType
from backend.app.models.fusion.ablation import (
    ABLATION_SUITE,
    AblationEvaluator,
)
from backend.app.models.fusion.evidence_fusion import EvidenceAwareFusion
from backend.app.multimodal.vision_encoder import MultimodalConsistencyResult


def test_fusion_insufficient_evidence_gate():
    """Verify epistemic gate suppresses binary classification when evidence sufficiency < 0.40."""
    out = EvidenceAwareFusion.fuse(
        text_probabilities=[0.35, 0.35, 0.30],
        evidence_items=[],  # Zero evidence
        multimodal_result=None,
    )
    assert out.state == CredibilityState.INSUFFICIENT_EVIDENCE
    assert out.evidence_sufficiency == 0.0
    assert out.calibrated_confidence <= 0.40
    assert "Insufficient evidence" in out.decision_rationale


def test_fusion_refutation_dominance():
    """Verify that strong contradictory evidence from reliable sources dictates LIKELY_FALSE."""
    evid = [
        RetrievedEvidence(
            evidence_id="e1",
            claim_id="c1",
            source_domain="reuters.com",
            source_url="https://reuters.com/factcheck",
            title="Debunked",
            snippet="Claims are false and directly refuted by public documentation.",
            relevance_score=0.92,
            stance=StanceType.CONTRADICTS,
            stance_confidence=0.95,
            retrieval_provider="Reuters",
            reliability_score=0.96,
        )
    ]
    out = EvidenceAwareFusion.fuse(
        text_probabilities=[0.40, 0.30, 0.30],
        evidence_items=evid,
    )
    assert out.state == CredibilityState.LIKELY_FALSE
    assert out.evidence_sufficiency >= 0.40
    assert out.calibrated_confidence >= 0.70


def test_fusion_corroboration_dominance():
    """Verify that strong supporting evidence dictates LIKELY_TRUE."""
    evid = [
        RetrievedEvidence(
            evidence_id="e2",
            claim_id="c2",
            source_domain="apnews.com",
            source_url="https://apnews.com/article",
            title="Verified",
            snippet="Official reports confirmed and substantiated the reported numbers.",
            relevance_score=0.90,
            stance=StanceType.SUPPORTS,
            stance_confidence=0.92,
            retrieval_provider="AP",
            reliability_score=0.95,
        )
    ]
    out = EvidenceAwareFusion.fuse(
        text_probabilities=[0.20, 0.50, 0.30],
        evidence_items=evid,
    )
    assert out.state == CredibilityState.LIKELY_TRUE
    assert out.evidence_sufficiency >= 0.40
    assert out.calibrated_confidence >= 0.70


def test_fusion_conflict_arbitration():
    """Verify that conflicting evidence triggers MISLEADING_OR_PARTIALLY_SUPPORTED."""
    evid = [
        RetrievedEvidence(
            evidence_id="e3a",
            claim_id="c3",
            source_domain="politifact.com",
            source_url="https://politifact.com",
            title="PolitiFact",
            snippet="Official records confirm part of the claim is true.",
            relevance_score=0.88,
            stance=StanceType.SUPPORTS,
            stance_confidence=0.80,
            retrieval_provider="PolitiFact",
            reliability_score=0.94,
        ),
        RetrievedEvidence(
            evidence_id="e3b",
            claim_id="c3",
            source_domain="snopes.com",
            source_url="https://snopes.com",
            title="Snopes",
            snippet="Contextual figures directly contradict the author's causal claims.",
            relevance_score=0.88,
            stance=StanceType.CONTRADICTS,
            stance_confidence=0.80,
            retrieval_provider="Snopes",
            reliability_score=0.92,
        ),
    ]
    out = EvidenceAwareFusion.fuse(
        text_probabilities=[0.33, 0.33, 0.34],
        evidence_items=evid,
    )
    assert out.state == CredibilityState.MISLEADING_OR_PARTIALLY_SUPPORTED
    assert out.conflict_score >= 0.40


def test_fusion_visual_mismatch_penalty():
    """Verify that out-of-context visual mismatch shifts otherwise supported text."""
    mm_mismatch = MultimodalConsistencyResult(
        has_image=True,
        alignment_score=0.20,
        consistency_state="MISMATCH_SUSPECTED",
        visual_features=[],
        rationale="Graphic text does not match claim subject matter.",
    )
    out = EvidenceAwareFusion.fuse(
        text_probabilities=[0.10, 0.80, 0.10],  # Text alone looks true
        evidence_items=None,
        multimodal_result=mm_mismatch,
        include_evidence=False,  # Test text + image ablation
    )
    assert out.state == CredibilityState.MISLEADING_OR_PARTIALLY_SUPPORTED
    assert "cross-modal mismatch" in out.decision_rationale


def test_ablation_suite_execution():
    """Verify that all 9 ablation configurations execute and produce structured results."""
    results = AblationEvaluator.run_benchmark_ablation_suite()
    assert len(results) == len(ABLATION_SUITE)
    config_names = [r["config_name"] for r in results]
    assert "full_model" in config_names
    assert "text_only" in config_names
    assert "minus_evidence" in config_names

    for r in results:
        assert 0.0 <= r["accuracy"] <= 1.0
        assert 0.0 <= r["mean_calibrated_confidence"] <= 1.0
