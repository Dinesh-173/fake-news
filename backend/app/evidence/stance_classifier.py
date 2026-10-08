"""TruthLens Stance Classifier and Evidence Sufficiency Engine.

Evaluates the Natural Language Inference (NLI) relationship between claim-evidence pairs:
- SUPPORTS
- CONTRADICTS
- NEUTRAL

Computes:
1. Aggregate Support (S_supp) and Contradiction (S_contra) scores.
2. Mathematical Conflict Index: 2 * min(S_supp, S_contra).
3. Evidence Sufficiency: ratio of relevant non-neutral evidence to threshold.
4. Final Evidentiary Verdict: SUPPORTED, REFUTED, MIXED, INSUFFICIENT_EVIDENCE.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from backend.app.models.entities import StanceType

REFUTATION_KEYWORDS = {
    "false", "fake", "debunked", "debunk", "hoax", "incorrect", "untrue",
    "misleading", "fabricated", "unfounded", "denied", "disproved", "disproven",
    "no evidence", "never happened", "refuted", "inaccurate", "distorted",
    "pants on fire", "myth", "baseless", "contradicts", "false claim",
}

SUPPORT_KEYWORDS = {
    "confirmed", "verified", "accurate", "true", "authentic", "corroborated",
    "substantiated", "official report", "evidence shows", "documented",
    "certified", "validated", "proven", "proves", "endorsed", "confirmed by",
}


@dataclass
class StanceResult:
    stance: StanceType
    confidence: float
    support_score: float
    contradiction_score: float
    neutral_score: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stance": self.stance.value,
            "confidence": round(self.confidence, 3),
            "support_score": round(self.support_score, 3),
            "contradiction_score": round(self.contradiction_score, 3),
            "neutral_score": round(self.neutral_score, 3),
            "rationale": self.rationale,
        }


@dataclass
class EvidenceSufficiencyProfile:
    sufficiency_score: float
    conflict_score: float
    mean_support: float
    mean_contradiction: float
    verdict: str  # SUPPORTED, REFUTED, MIXED, INSUFFICIENT_EVIDENCE
    total_evidence_count: int
    active_evidence_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficiency_score": round(self.sufficiency_score, 3),
            "conflict_score": round(self.conflict_score, 3),
            "mean_support": round(self.mean_support, 3),
            "mean_contradiction": round(self.mean_contradiction, 3),
            "verdict": self.verdict,
            "total_evidence_count": self.total_evidence_count,
            "active_evidence_count": self.active_evidence_count,
        }


class StanceClassifier:
    """Classifies claim-evidence stances and calculates consensus/conflict metrics."""

    @classmethod
    def classify_stance(cls, claim_text: str, evidence_snippet: str) -> StanceResult:
        """
        Evaluates the stance of an evidence passage toward a specific claim.
        Employs lexical-semantic NLI scoring with negation and refutation detection.
        """
        c_lower = claim_text.lower()
        e_lower = evidence_snippet.lower()

        # Count refutation cues
        contra_hits = sum(1 for kw in REFUTATION_KEYWORDS if kw in e_lower)
        # Count affirmation/support cues
        supp_hits = sum(1 for kw in SUPPORT_KEYWORDS if kw in e_lower)

        # Check for direct debunking phrases
        is_explicit_debunk = bool(
            re.search(r'\b(?:is false|are false|claim is false|no evidence that|untrue that|did not happen)\b', e_lower)
        )
        if is_explicit_debunk:
            contra_hits += 3

        # Token overlap between claim and evidence
        c_tokens = set(re.findall(r'\b\w{4,}\b', c_lower))
        e_tokens = set(re.findall(r'\b\w{4,}\b', e_lower))
        overlap = len(c_tokens.intersection(e_tokens)) / max(1, len(c_tokens))

        # Decision boundaries
        if contra_hits > supp_hits and contra_hits >= 1 and overlap >= 0.20:
            contra_score = min(0.98, 0.65 + (contra_hits * 0.10))
            supp_score = 0.05
            neut_score = round(1.0 - contra_score - supp_score, 3)
            return StanceResult(
                stance=StanceType.CONTRADICTS,
                confidence=contra_score,
                support_score=supp_score,
                contradiction_score=contra_score,
                neutral_score=max(0.0, neut_score),
                rationale=f"Evidence directly refutes claim assertions ({contra_hits} refutation markers).",
            )
        elif supp_hits > contra_hits and supp_hits >= 1 and overlap >= 0.25:
            supp_score = min(0.98, 0.65 + (supp_hits * 0.10))
            contra_score = 0.05
            neut_score = round(1.0 - supp_score - contra_score, 3)
            return StanceResult(
                stance=StanceType.SUPPORTS,
                confidence=supp_score,
                support_score=supp_score,
                contradiction_score=contra_score,
                neutral_score=max(0.0, neut_score),
                rationale=f"Evidence corroborates claim assertions ({supp_hits} support markers).",
            )
        else:
            neut_score = 0.70
            supp_score = 0.15
            contra_score = 0.15
            return StanceResult(
                stance=StanceType.NEUTRAL,
                confidence=neut_score,
                support_score=supp_score,
                contradiction_score=contra_score,
                neutral_score=neut_score,
                rationale="Passage discusses related context but provides neutral stance or insufficient direct affirmation/refutation.",
            )

    @classmethod
    def evaluate_sufficiency_and_consensus(
        cls,
        evidence_evaluations: list[dict[str, Any]],
        sufficiency_threshold_k: float = 2.0,
    ) -> EvidenceSufficiencyProfile:
        """
        Aggregates multiple evidence evaluations to compute overall sufficiency,
        conflict score, and verdict.

        Formula:
          Sufficiency = min(1.0, sum(Relevance_j * 1[Stance_j != NEUTRAL]) / K_threshold)
          Conflict Score = 2 * min(Mean_Support, Mean_Contradiction)
        """
        if not evidence_evaluations:
            return EvidenceSufficiencyProfile(
                sufficiency_score=0.0,
                conflict_score=0.0,
                mean_support=0.0,
                mean_contradiction=0.0,
                verdict="INSUFFICIENT_EVIDENCE",
                total_evidence_count=0,
                active_evidence_count=0,
            )

        total_count = len(evidence_evaluations)
        weighted_support = 0.0
        weighted_contra = 0.0
        active_weight_sum = 0.0
        effective_evidence_signal = 0.0

        for item in evidence_evaluations:
            rel = float(item.get("relevance_score", 1.0))
            stance = item.get("stance")
            stance_type = stance if isinstance(stance, StanceType) else StanceType(str(stance))
            supp = float(item.get("support_score", 0.0))
            contra = float(item.get("contradiction_score", 0.0))
            reliability = float(item.get("reliability_score", 0.50))

            weight = rel * reliability

            if stance_type != StanceType.NEUTRAL:
                effective_evidence_signal += rel * reliability
                weighted_support += supp * weight
                weighted_contra += contra * weight
                active_weight_sum += weight

        sufficiency = min(1.0, effective_evidence_signal / max(0.1, sufficiency_threshold_k))

        if active_weight_sum > 0:
            mean_supp = weighted_support / active_weight_sum
            mean_contra = weighted_contra / active_weight_sum
        else:
            mean_supp = 0.0
            mean_contra = 0.0

        conflict_score = min(1.0, 2.0 * min(mean_supp, mean_contra))

        # Verdict logic
        if sufficiency < 0.40:
            verdict = "INSUFFICIENT_EVIDENCE"
        elif conflict_score >= 0.40:
            verdict = "MIXED"
        elif mean_contra > 0.55 and mean_supp < 0.30:
            verdict = "REFUTED"
        elif mean_supp > 0.55 and mean_contra < 0.30:
            verdict = "SUPPORTED"
        else:
            verdict = "MIXED"

        return EvidenceSufficiencyProfile(
            sufficiency_score=sufficiency,
            conflict_score=conflict_score,
            mean_support=mean_supp,
            mean_contradiction=mean_contra,
            verdict=verdict,
            total_evidence_count=total_count,
            active_evidence_count=int(effective_evidence_signal),
        )
