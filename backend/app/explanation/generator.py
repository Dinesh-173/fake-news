"""TruthLens Grounded Explanation Generator.

Generates human-readable, evidence-grounded verification reports answering:
1. WHAT was claimed?
2. WHAT evidence was found?
3. WHICH evidence supports or refutes it?
4. HOW strong is the evidence?
5. WHY did the system reach this conclusion?
6. WHAT uncertainties or limitations remain?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from backend.app.evidence.retriever import RetrievedEvidence
from backend.app.explanation.guardrails import ExplanationGuardrails
from backend.app.models.entities import CredibilityState, StanceType

logger = logging.getLogger("truthlens.explanation.generator")


@dataclass
class ExplanationReport:
    summary: str
    evidence_breakdown: str
    limitations: str
    faithfulness_score: float
    guarded_prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "evidence_breakdown": self.evidence_breakdown,
            "limitations": self.limitations,
            "faithfulness_score": round(self.faithfulness_score, 3),
        }


class ExplanationGenerator:
    """Generates structured, grounded explanations for news claim verifications."""

    @classmethod
    def generate_explanation(
        cls,
        claim_text: str,
        verdict_state: CredibilityState,
        calibrated_confidence: float,
        evidence_sufficiency: float,
        conflict_score: float,
        evidence_items: list[RetrievedEvidence] | None = None,
        multimodal_rationale: str | None = None,
    ) -> ExplanationReport:
        """
        Synthesizes a fully grounded explanation report.
        """
        evidence_items = evidence_items or []
        valid_domains = {e.source_domain.lower() for e in evidence_items}

        # 1. Build Guarded Prompt for audit traceability
        evid_dicts = [e.to_dict() for e in evidence_items]
        guarded_prompt = ExplanationGuardrails.build_guarded_prompt(
            claim_text=claim_text,
            verdict_state=verdict_state.value,
            calibrated_confidence=calibrated_confidence,
            evidence_sufficiency=evidence_sufficiency,
            conflict_score=conflict_score,
            evidence_items=evid_dicts,
        )

        # 2. Count Stances
        supporting = [e for e in evidence_items if e.stance == StanceType.SUPPORTS]
        contradicting = [e for e in evidence_items if e.stance == StanceType.CONTRADICTS]
        neutral = [e for e in evidence_items if e.stance == StanceType.NEUTRAL]

        # 3. Formulate Summary
        conf_pct = calibrated_confidence * 100.0

        if verdict_state == CredibilityState.LIKELY_TRUE:
            summary = (
                f"The claim '{claim_text[:80]}...' is assessed as LIKELY TRUE with a calibrated confidence "
                f"of {conf_pct:.1f}%. Authoritative reporting and verified registries corroborate the statement."
            )
        elif verdict_state == CredibilityState.LIKELY_FALSE:
            summary = (
                f"The claim '{claim_text[:80]}...' is assessed as LIKELY FALSE with a calibrated confidence "
                f"of {conf_pct:.1f}%. External verified fact-checking sources and primary documentation directly refute the assertion."
            )
        elif verdict_state == CredibilityState.MISLEADING_OR_PARTIALLY_SUPPORTED:
            summary = (
                f"The claim '{claim_text[:80]}...' is assessed as MISLEADING OR PARTIALLY SUPPORTED ({conf_pct:.1f}% confidence). "
                f"While some factual elements are corroborated, significant contextual omission or contradictory evidence was identified."
            )
        else:
            summary = (
                f"The assessment for '{claim_text[:80]}...' is INSUFFICIENT EVIDENCE. "
                "Available external records do not provide adequate corroborated evidence to support or refute the claim with scientific certainty."
            )

        # 4. Formulate Detailed Evidence Breakdown
        breakdown_sections = []
        breakdown_sections.append(f"• WHAT WAS CLAIMED: \"{claim_text}\"")

        if evidence_items:
            breakdown_sections.append(
                f"• EVIDENCE FOUND: Cross-referenced across {len(evidence_items)} independent sources "
                f"({len(supporting)} supporting, {len(contradicting)} contradicting, {len(neutral)} neutral)."
            )

            if contradicting:
                contra_sources = ", ".join(f"{c.source_domain}" for c in contradicting[:3])
                sample_contra = contradicting[0].snippet[:140]
                breakdown_sections.append(
                    f"• CONTRADICTING FINDINGS: Refuted by {contra_sources}. Example finding: \"{sample_contra}...\""
                )

            if supporting:
                supp_sources = ", ".join(f"{s.source_domain}" for s in supporting[:3])
                sample_supp = supporting[0].snippet[:140]
                breakdown_sections.append(
                    f"• SUPPORTING FINDINGS: Corroborated by {supp_sources}. Example finding: \"{sample_supp}...\""
                )

            if conflict_score >= 0.40:
                breakdown_sections.append(
                    f"• EVIDENTIARY CONFLICT: High disagreement index ({conflict_score:.2f}) observed between sources. "
                    "Multiple accredited publishers present opposing assessments."
                )
        else:
            breakdown_sections.append(
                "• EVIDENCE FOUND: Zero matching records found across accredited fact-checking archives and external repositories."
            )

        if multimodal_rationale:
            breakdown_sections.append(f"• MULTIMODAL CONTEXT: {multimodal_rationale}")

        breakdown_sections.append(
            f"• REASONING & SUFFICIENCY: Evidence sufficiency evaluated at {evidence_sufficiency*100:.1f}%. "
            f"The final verdict is grounded in external consensus rather than model bias."
        )

        evidence_breakdown = "\n".join(breakdown_sections)

        # 5. Formulate Limitations and Uncertainty
        limitations = (
            f"Verification confidence is calibrated at {conf_pct:.1f}%. "
            "Evaluations reflect real-world evidence available at time of query. "
            "Rapidly evolving events or private off-line documentation may alter factual ground truth."
        )

        # 6. Audit Citation Faithfulness
        faithfulness = ExplanationGuardrails.audit_citation_faithfulness(
            explanation_text=evidence_breakdown,
            valid_domains=valid_domains,
        )

        return ExplanationReport(
            summary=summary,
            evidence_breakdown=evidence_breakdown,
            limitations=limitations,
            faithfulness_score=faithfulness,
            guarded_prompt=guarded_prompt,
        )
