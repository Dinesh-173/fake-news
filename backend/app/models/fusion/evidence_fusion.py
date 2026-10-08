"""TruthLens Evidence-Aware Multimodal Fusion Engine.

Combines:
1. Textual claim representation (h_text).
2. Visual representation & cross-modal consistency (h_image).
3. Retrieved evidence stance & conflict vectors (h_evid).
4. Source credibility and provenance features (h_source).
5. Claim metadata and proposition salience (h_meta).

Enforces:
- Epistemic Degradation Gate: suppresses forced binary verdicts when evidence sufficiency < tau_suff.
- Conflict Arbitration Gate: flags MISLEADING_OR_PARTIALLY_SUPPORTED when evidentiary conflict > tau_conflict.
- Veracity Grounding: prioritizes verified external evidence over ungrounded internal model bias.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from backend.app.evidence.retriever import RetrievedEvidence
from backend.app.evidence.stance_classifier import (
    EvidenceSufficiencyProfile,
    StanceClassifier,
)
from backend.app.models.entities import CredibilityState, StanceType
from backend.app.multimodal.vision_encoder import MultimodalConsistencyResult

logger = logging.getLogger("truthlens.models.fusion")


@dataclass
class FusionOutput:
    state: CredibilityState
    prediction_confidence: float
    calibrated_confidence: float
    evidence_sufficiency: float
    conflict_score: float
    alignment_score: float
    contributing_signals: dict[str, float]
    decision_rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "prediction_confidence": round(self.prediction_confidence, 3),
            "calibrated_confidence": round(self.calibrated_confidence, 3),
            "evidence_sufficiency": round(self.evidence_sufficiency, 3),
            "conflict_score": round(self.conflict_score, 3),
            "alignment_score": round(self.alignment_score, 3),
            "contributing_signals": {k: round(v, 3) for k, v in self.contributing_signals.items()},
            "decision_rationale": self.decision_rationale,
        }


class EvidenceAwareFusion:
    """Core multimodal evidence fusion engine with epistemic uncertainty gates."""

    SUFFICIENCY_THRESHOLD: float = 0.40
    CONFLICT_THRESHOLD: float = 0.40

    @classmethod
    def fuse(
        cls,
        text_probabilities: list[float] | None,  # [P(FALSE), P(TRUE), P(HALF)]
        evidence_items: list[RetrievedEvidence] | None,
        multimodal_result: MultimodalConsistencyResult | None = None,
        claim_salience: float = 1.0,
        include_image: bool = True,
        include_evidence: bool = True,
        include_source: bool = True,
        include_metadata: bool = True,
    ) -> FusionOutput:
        """
        Executes principled multimodal evidence fusion with modular ablation toggles.
        """
        # 1. Text Signal Extraction
        # Canonical indices: 0 -> LIKELY_FALSE, 1 -> LIKELY_TRUE, 2 -> MISLEADING_OR_PARTIALLY_SUPPORTED
        if text_probabilities and len(text_probabilities) == 3:
            p_false = float(text_probabilities[0])
            p_true = float(text_probabilities[1])
            p_misleading = float(text_probabilities[2])
        else:
            p_false, p_true, p_misleading = 0.333, 0.333, 0.334

        # 2. Evidence Signal Extraction
        if include_evidence and evidence_items:
            evid_dicts = [
                {
                    "relevance_score": e.relevance_score,
                    "stance": e.stance,
                    "support_score": 0.85 if e.stance == StanceType.SUPPORTS else 0.05,
                    "contradiction_score": 0.85 if e.stance == StanceType.CONTRADICTS else 0.05,
                    "reliability_score": e.reliability_score if include_source else 0.50,
                }
                for e in evidence_items
            ]
            sufficiency_profile = StanceClassifier.evaluate_sufficiency_and_consensus(evid_dicts)
        else:
            sufficiency_profile = EvidenceSufficiencyProfile(
                sufficiency_score=0.0,
                conflict_score=0.0,
                mean_support=0.0,
                mean_contradiction=0.0,
                verdict="INSUFFICIENT_EVIDENCE",
                total_evidence_count=0,
                active_evidence_count=0,
            )

        # 3. Multimodal Signal Extraction
        if include_image and multimodal_result and multimodal_result.has_image:
            vis_alignment = multimodal_result.alignment_score
            has_image_flag = 1.0
        else:
            vis_alignment = 1.0
            has_image_flag = 0.0

        # 4. Source & Metadata Signals
        if include_source and evidence_items:
            mean_source_reliability = float(
                sum(e.reliability_score for e in evidence_items) / len(evidence_items)
            )
            has_fact_checker = 1.0 if any(e.reliability_score >= 0.90 for e in evidence_items) else 0.0
        else:
            mean_source_reliability = 0.50
            has_fact_checker = 0.0

        meta_salience = claim_salience if include_metadata else 1.0

        # Signals dictionary for transparency and ablation auditing
        signals = {
            "text_p_false": p_false,
            "text_p_true": p_true,
            "text_p_misleading": p_misleading,
            "evidence_sufficiency": sufficiency_profile.sufficiency_score,
            "evidence_conflict": sufficiency_profile.conflict_score,
            "evidence_support": sufficiency_profile.mean_support,
            "evidence_contradiction": sufficiency_profile.mean_contradiction,
            "visual_alignment": vis_alignment,
            "has_image": has_image_flag,
            "source_reliability": mean_source_reliability,
            "fact_checker_present": has_fact_checker,
            "claim_salience": meta_salience,
        }

        # 5. Core Decision Logic with Safety Gates

        # GATE 1: Insufficient Evidence Check
        # If evidence was included but sufficiency score is below threshold,
        # suppress forced binary classification and return INSUFFICIENT_EVIDENCE.
        if include_evidence and sufficiency_profile.sufficiency_score < cls.SUFFICIENCY_THRESHOLD:
            return FusionOutput(
                state=CredibilityState.INSUFFICIENT_EVIDENCE,
                prediction_confidence=0.50,
                calibrated_confidence=0.35,
                evidence_sufficiency=sufficiency_profile.sufficiency_score,
                conflict_score=sufficiency_profile.conflict_score,
                alignment_score=vis_alignment,
                contributing_signals=signals,
                decision_rationale="Insufficient evidence retrieved (<40% sufficiency threshold). Withholding definitive veracity verdict to prevent ungrounded prediction.",
            )

        # GATE 2: Evidence Consensus Decision
        if include_evidence and sufficiency_profile.sufficiency_score >= cls.SUFFICIENCY_THRESHOLD:
            # Check for high evidentiary conflict
            if sufficiency_profile.conflict_score >= cls.CONFLICT_THRESHOLD:
                conf = round(0.60 + 0.30 * sufficiency_profile.sufficiency_score, 3)
                return FusionOutput(
                    state=CredibilityState.MISLEADING_OR_PARTIALLY_SUPPORTED,
                    prediction_confidence=conf,
                    calibrated_confidence=round(conf * 0.90, 3),
                    evidence_sufficiency=sufficiency_profile.sufficiency_score,
                    conflict_score=sufficiency_profile.conflict_score,
                    alignment_score=vis_alignment,
                    contributing_signals=signals,
                    decision_rationale=f"Significant evidentiary contradiction detected (Conflict Index: {sufficiency_profile.conflict_score:.2f}). Both supporting and refuting sources observed.",
                )

            # Check for dominant contradiction / debunking
            if sufficiency_profile.mean_contradiction > 0.50:
                conf = min(
                    0.98,
                    round(
                        0.70
                        + 0.20 * sufficiency_profile.mean_contradiction
                        + 0.10 * mean_source_reliability,
                        3,
                    ),
                )
                return FusionOutput(
                    state=CredibilityState.LIKELY_FALSE,
                    prediction_confidence=conf,
                    calibrated_confidence=round(conf * mean_source_reliability, 3),
                    evidence_sufficiency=sufficiency_profile.sufficiency_score,
                    conflict_score=sufficiency_profile.conflict_score,
                    alignment_score=vis_alignment,
                    contributing_signals=signals,
                    decision_rationale=f"Primary claims refuted by external verified sources (Mean Contradiction: {sufficiency_profile.mean_contradiction:.2f}).",
                )

            # Check for dominant support / corroboration
            if sufficiency_profile.mean_support > 0.50:
                conf = min(
                    0.98,
                    round(
                        0.70
                        + 0.20 * sufficiency_profile.mean_support
                        + 0.10 * mean_source_reliability,
                        3,
                    ),
                )
                return FusionOutput(
                    state=CredibilityState.LIKELY_TRUE,
                    prediction_confidence=conf,
                    calibrated_confidence=round(conf * mean_source_reliability, 3),
                    evidence_sufficiency=sufficiency_profile.sufficiency_score,
                    conflict_score=sufficiency_profile.conflict_score,
                    alignment_score=vis_alignment,
                    contributing_signals=signals,
                    decision_rationale=f"Core claims corroborated by authoritative external reporting (Mean Support: {sufficiency_profile.mean_support:.2f}).",
                )

        # GATE 3: Fallback / Text-Only Ablation Mode
        if p_false >= max(p_true, p_misleading):
            state = CredibilityState.LIKELY_FALSE
            conf = p_false
        elif p_true >= max(p_false, p_misleading):
            state = CredibilityState.LIKELY_TRUE
            conf = p_true
        else:
            state = CredibilityState.MISLEADING_OR_PARTIALLY_SUPPORTED
            conf = p_misleading

        # Adjust for visual mismatch if image is present
        if include_image and multimodal_result and multimodal_result.consistency_state == "MISMATCH_SUSPECTED":
            # Potential out-of-context image reuse slightly shifts toward misleading
            if state == CredibilityState.LIKELY_TRUE:
                state = CredibilityState.MISLEADING_OR_PARTIALLY_SUPPORTED
                rationale = "Textual claims appear supported, but embedded image exhibits substantial cross-modal mismatch (possible out-of-context graphic)."
            else:
                rationale = "Assessment guided by text representation and potential visual-semantic context mismatch."
        else:
            rationale = "Assessment derived from fused textual and contextual priors."

        return FusionOutput(
            state=state,
            prediction_confidence=conf,
            calibrated_confidence=round(conf * 0.85, 3),
            evidence_sufficiency=sufficiency_profile.sufficiency_score,
            conflict_score=sufficiency_profile.conflict_score,
            alignment_score=vis_alignment,
            contributing_signals=signals,
            decision_rationale=rationale,
        )
