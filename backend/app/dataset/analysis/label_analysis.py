"""Label distribution and mapping coverage analysis module."""

import math
from collections import Counter

from pydantic import BaseModel, Field

from backend.app.dataset.schemas import CanonicalNewsRecord


class LabelMappingBreakdown(BaseModel):
    """Detailed provenance row for an individual label mapping."""

    original_label: str
    normalized_label: str | None
    status: str
    count: int
    percentage: float


class LabelAnalysisResult(BaseModel):
    """Encapsulates comprehensive label distribution and class balance metrics."""

    total_records: int
    unique_original_labels: int
    unique_normalized_labels: int
    original_distribution: dict[str, int]
    original_percentages: dict[str, float]
    normalized_distribution: dict[str, int]
    normalized_percentages: dict[str, float]
    mapping_breakdown: list[LabelMappingBreakdown]
    mapping_coverage: float = Field(..., description="Ratio of records mapped to a canonical label")
    unresolved_count: int
    unresolved_labels: list[str]
    majority_class: str
    majority_class_percentage: float
    minority_class: str
    minority_class_percentage: float
    imbalance_ratio: float = Field(..., description="Majority count / Minority count")
    normalized_entropy: float = Field(..., description="Shannon entropy normalized to [0, 1]")


class LabelAnalyzer:
    """Computes comprehensive label distribution, mapping coverage, and class imbalance metrics."""

    @classmethod
    def analyze(cls, records: list[CanonicalNewsRecord]) -> LabelAnalysisResult:
        if not records:
            return LabelAnalysisResult(
                total_records=0,
                unique_original_labels=0,
                unique_normalized_labels=0,
                original_distribution={},
                original_percentages={},
                normalized_distribution={},
                normalized_percentages={},
                mapping_breakdown=[],
                mapping_coverage=0.0,
                unresolved_count=0,
                unresolved_labels=[],
                majority_class="NONE",
                majority_class_percentage=0.0,
                minority_class="NONE",
                minority_class_percentage=0.0,
                imbalance_ratio=1.0,
                normalized_entropy=0.0,
            )

        total = len(records)
        orig_counter = Counter(r.original_label for r in records)
        norm_counter = Counter(r.normalized_label or "UNRESOLVED" for r in records)

        orig_pct = {k: round((v / total) * 100, 2) for k, v in orig_counter.items()}
        norm_pct = {k: round((v / total) * 100, 2) for k, v in norm_counter.items()}

        # Build detailed mapping breakdown
        pair_counter = Counter(
            (r.original_label, r.normalized_label, r.label_mapping_status) for r in records
        )
        mapping_breakdown: list[LabelMappingBreakdown] = []
        for (orig, norm, status), count in sorted(
            pair_counter.items(), key=lambda x: x[1], reverse=True
        ):
            mapping_breakdown.append(
                LabelMappingBreakdown(
                    original_label=orig,
                    normalized_label=norm,
                    status=status,
                    count=count,
                    percentage=round((count / total) * 100, 2),
                )
            )

        unresolved_recs = [
            r for r in records if not r.normalized_label or r.label_mapping_status != "mapped"
        ]
        unresolved_labels = sorted({r.original_label for r in unresolved_recs})
        unresolved_count = len(unresolved_recs)
        mapped_count = total - unresolved_count
        mapping_coverage = round(mapped_count / total, 4)

        # Imbalance calculations on normalized distribution (excluding UNRESOLVED if other labels exist)
        eval_counter = {k: v for k, v in norm_counter.items() if k != "UNRESOLVED"}
        if not eval_counter:
            eval_counter = norm_counter

        sorted_classes = sorted(eval_counter.items(), key=lambda x: x[1], reverse=True)
        majority_class, maj_count = sorted_classes[0]
        minority_class, min_count = sorted_classes[-1]
        maj_pct = round((maj_count / total) * 100, 2)
        min_pct = round((min_count / total) * 100, 2)
        imbalance_ratio = round(maj_count / max(min_count, 1), 2)

        # Normalized Shannon entropy: H = -sum(p * log2(p)) / log2(K)
        k = len(sorted_classes)
        if k > 1:
            h = -sum(
                (cnt / total) * math.log2(cnt / total) for _, cnt in sorted_classes if cnt > 0
            )
            norm_entropy = round(h / math.log2(k), 4)
        else:
            norm_entropy = 0.0

        return LabelAnalysisResult(
            total_records=total,
            unique_original_labels=len(orig_counter),
            unique_normalized_labels=len(eval_counter),
            original_distribution=dict(orig_counter),
            original_percentages=orig_pct,
            normalized_distribution=dict(norm_counter),
            normalized_percentages=norm_pct,
            mapping_breakdown=mapping_breakdown,
            mapping_coverage=mapping_coverage,
            unresolved_count=unresolved_count,
            unresolved_labels=unresolved_labels,
            majority_class=majority_class,
            majority_class_percentage=maj_pct,
            minority_class=minority_class,
            minority_class_percentage=min_pct,
            imbalance_ratio=imbalance_ratio,
            normalized_entropy=norm_entropy,
        )
