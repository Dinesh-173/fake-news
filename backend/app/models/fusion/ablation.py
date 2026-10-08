"""TruthLens Systematic Ablation Study Framework.

Evaluates contribution of individual modality components:
1. Full Multimodal Evidence-Aware TruthLens
2. Text-only
3. Text + Image
4. Text + Evidence
5. Text + Source/Metadata
6. Minus Image
7. Minus Evidence
8. Minus Source
9. Minus Metadata
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.app.evidence.retriever import RetrievedEvidence
from backend.app.models.entities import CredibilityState, StanceType
from backend.app.models.fusion.evidence_fusion import (
    EvidenceAwareFusion,
    FusionOutput,
)
from backend.app.multimodal.vision_encoder import MultimodalConsistencyResult


@dataclass
class AblationConfiguration:
    config_name: str
    description: str
    include_image: bool = True
    include_evidence: bool = True
    include_source: bool = True
    include_metadata: bool = True


ABLATION_SUITE: list[AblationConfiguration] = [
    AblationConfiguration(
        config_name="full_model",
        description="Full Multimodal Evidence-Aware Framework (Text + Image + Evidence + Source + Metadata)",
        include_image=True,
        include_evidence=True,
        include_source=True,
        include_metadata=True,
    ),
    AblationConfiguration(
        config_name="text_only",
        description="Text Classifier Only (No external evidence, no image, no source credibility)",
        include_image=False,
        include_evidence=False,
        include_source=False,
        include_metadata=False,
    ),
    AblationConfiguration(
        config_name="text_plus_image",
        description="Text + Image (No external retrieval, no source credibility)",
        include_image=True,
        include_evidence=False,
        include_source=False,
        include_metadata=False,
    ),
    AblationConfiguration(
        config_name="text_plus_evidence",
        description="Text + Evidence (Unweighted external evidence, no image, no source credibility)",
        include_image=False,
        include_evidence=True,
        include_source=False,
        include_metadata=False,
    ),
    AblationConfiguration(
        config_name="text_plus_source_meta",
        description="Text + Source/Metadata (No image, no retrieved evidence passages)",
        include_image=False,
        include_evidence=False,
        include_source=True,
        include_metadata=True,
    ),
    AblationConfiguration(
        config_name="minus_image",
        description="Ablation: Full Framework excluding Image Analysis",
        include_image=False,
        include_evidence=True,
        include_source=True,
        include_metadata=True,
    ),
    AblationConfiguration(
        config_name="minus_evidence",
        description="Ablation: Full Framework excluding Evidence Retrieval",
        include_image=True,
        include_evidence=False,
        include_source=True,
        include_metadata=True,
    ),
    AblationConfiguration(
        config_name="minus_source",
        description="Ablation: Full Framework excluding Source Credibility",
        include_image=True,
        include_evidence=True,
        include_source=False,
        include_metadata=True,
    ),
    AblationConfiguration(
        config_name="minus_metadata",
        description="Ablation: Full Framework excluding Claim Metadata & Salience",
        include_image=True,
        include_evidence=True,
        include_source=True,
        include_metadata=False,
    ),
]


class AblationEvaluator:
    """Runs ablation experiments and calculates component contributions."""

    @classmethod
    def evaluate_case(
        cls,
        text_probs: list[float],
        evidence_items: list[RetrievedEvidence] | None,
        multimodal_result: MultimodalConsistencyResult | None,
        claim_salience: float = 1.0,
        config: AblationConfiguration | None = None,
    ) -> FusionOutput:
        """Evaluates a single sample under a specific ablation configuration."""
        if config is None:
            config = ABLATION_SUITE[0]

        return EvidenceAwareFusion.fuse(
            text_probabilities=text_probs,
            evidence_items=evidence_items,
            multimodal_result=multimodal_result,
            claim_salience=claim_salience,
            include_image=config.include_image,
            include_evidence=config.include_evidence,
            include_source=config.include_source,
            include_metadata=config.include_metadata,
        )

    @classmethod
    def run_benchmark_ablation_suite(cls) -> list[dict[str, Any]]:
        """
        Executes standard benchmark evaluation scenarios across all ablation configurations.
        Scenarios represent diverse real-world informational regimes:
        1. Debunked false claim with high-authority refutation.
        2. Verified true claim with corroboration.
        3. Conflicting / contested claim with mixed evidence.
        4. Novel / unverified claim with zero external evidence.
        5. Manipulated context claim with visual-semantic mismatch.
        """
        # Define canonical test scenarios
        sample_scenarios = [
            # Scenario 1: Refuted false claim
            {
                "ground_truth": CredibilityState.LIKELY_FALSE,
                "text_probs": [0.65, 0.20, 0.15],
                "evidence": [
                    RetrievedEvidence(
                        evidence_id="e1",
                        claim_id="c1",
                        source_domain="reuters.com",
                        source_url="https://reuters.com/factcheck",
                        title="Fact Check: Refuted",
                        snippet="Official documentation proves claim is false and unfounded.",
                        relevance_score=0.90,
                        stance=StanceType.CONTRADICTS,
                        stance_confidence=0.95,
                        retrieval_provider="Reuters",
                        reliability_score=0.96,
                    )
                ],
                "mm": MultimodalConsistencyResult(
                    has_image=True,
                    alignment_score=0.80,
                    consistency_state="ALIGNED",
                    visual_features=[],
                    rationale="",
                ),
            },
            # Scenario 2: Corroborated true claim
            {
                "ground_truth": CredibilityState.LIKELY_TRUE,
                "text_probs": [0.15, 0.70, 0.15],
                "evidence": [
                    RetrievedEvidence(
                        evidence_id="e2",
                        claim_id="c2",
                        source_domain="apnews.com",
                        source_url="https://apnews.com/article",
                        title="AP: Verified",
                        snippet="Officials confirmed and substantiated the reported numbers.",
                        relevance_score=0.92,
                        stance=StanceType.SUPPORTS,
                        stance_confidence=0.92,
                        retrieval_provider="AP",
                        reliability_score=0.95,
                    )
                ],
                "mm": MultimodalConsistencyResult(
                    has_image=True,
                    alignment_score=0.75,
                    consistency_state="ALIGNED",
                    visual_features=[],
                    rationale="",
                ),
            },
            # Scenario 3: Mixed / misleading claim
            {
                "ground_truth": CredibilityState.MISLEADING_OR_PARTIALLY_SUPPORTED,
                "text_probs": [0.30, 0.35, 0.35],
                "evidence": [
                    RetrievedEvidence(
                        evidence_id="e3a",
                        claim_id="c3",
                        source_domain="politifact.com",
                        source_url="https://politifact.com",
                        title="PolitiFact: Half True",
                        snippet="Parts of the statement are accurate, but key context is omitted.",
                        relevance_score=0.88,
                        stance=StanceType.SUPPORTS,
                        stance_confidence=0.75,
                        retrieval_provider="PolitiFact",
                        reliability_score=0.94,
                    ),
                    RetrievedEvidence(
                        evidence_id="e3b",
                        claim_id="c3",
                        source_domain="snopes.com",
                        source_url="https://snopes.com",
                        title="Snopes: Context Missing",
                        snippet="Critics and data directly contradict the causal framing.",
                        relevance_score=0.85,
                        stance=StanceType.CONTRADICTS,
                        stance_confidence=0.80,
                        retrieval_provider="Snopes",
                        reliability_score=0.92,
                    ),
                ],
                "mm": MultimodalConsistencyResult(
                    has_image=False,
                    alignment_score=1.0,
                    consistency_state="NO_IMAGE",
                    visual_features=[],
                    rationale="",
                ),
            },
            # Scenario 4: Sparse / uncorroborated claim
            {
                "ground_truth": CredibilityState.INSUFFICIENT_EVIDENCE,
                "text_probs": [0.34, 0.33, 0.33],
                "evidence": [],
                "mm": None,
            },
            # Scenario 5: Visual mismatch out-of-context image
            {
                "ground_truth": CredibilityState.MISLEADING_OR_PARTIALLY_SUPPORTED,
                "text_probs": [0.25, 0.50, 0.25],
                "evidence": [
                    RetrievedEvidence(
                        evidence_id="e5",
                        claim_id="c5",
                        source_domain="who.int",
                        source_url="https://who.int",
                        title="WHO Report",
                        snippet="WHO documented seasonal health guidelines.",
                        relevance_score=0.60,
                        stance=StanceType.NEUTRAL,
                        stance_confidence=0.70,
                        retrieval_provider="WHO",
                        reliability_score=0.98,
                    )
                ],
                "mm": MultimodalConsistencyResult(
                    has_image=True,
                    alignment_score=0.25,
                    consistency_state="MISMATCH_SUSPECTED",
                    visual_features=[],
                    rationale="",
                ),
            },
        ]

        results = []
        for config in ABLATION_SUITE:
            correct = 0
            confidences = []
            sufficiencies = []

            for sc in sample_scenarios:
                out = cls.evaluate_case(
                    text_probs=sc["text_probs"],
                    evidence_items=sc["evidence"],
                    multimodal_result=sc["mm"],
                    config=config,
                )
                if out.state == sc["ground_truth"]:
                    correct += 1
                confidences.append(out.calibrated_confidence)
                sufficiencies.append(out.evidence_sufficiency)

            acc = float(correct) / len(sample_scenarios)
            mean_conf = float(sum(confidences) / len(confidences))
            mean_suff = float(sum(sufficiencies) / len(sufficiencies))

            results.append({
                "config_name": config.config_name,
                "description": config.description,
                "include_image": config.include_image,
                "include_evidence": config.include_evidence,
                "include_source": config.include_source,
                "include_metadata": config.include_metadata,
                "accuracy": round(acc, 4),
                "mean_calibrated_confidence": round(mean_conf, 4),
                "mean_evidence_sufficiency": round(mean_suff, 4),
            })

        return results

    @classmethod
    def save_ablation_results(
        cls,
        output_dir: Path | str = "experiments/ablations",
    ) -> Path:
        """Runs and serializes the empirical ablation study to JSON and Markdown."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        results = cls.run_benchmark_ablation_suite()

        json_file = out_path / "ablation_results.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        md_file = out_path / "ablation_results.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# TruthLens: Empirical Ablation Study Results\n\n")
            f.write("| Configuration | Description | Image | Evidence | Source | Meta | Accuracy | Mean Confidence | Mean Sufficiency |\n")
            f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for r in results:
                f.write(
                    f"| `{r['config_name']}` | {r['description']} | "
                    f"{'✓' if r['include_image'] else '✗'} | "
                    f"{'✓' if r['include_evidence'] else '✗'} | "
                    f"{'✓' if r['include_source'] else '✗'} | "
                    f"{'✓' if r['include_metadata'] else '✗'} | "
                    f"**{r['accuracy']:.4f}** | {r['mean_calibrated_confidence']:.4f} | {r['mean_evidence_sufficiency']:.4f} |\n"
                )

        return json_file
