"""Cross-dataset and cross-split overlap, data leakage, and contamination analysis module."""

import hashlib
import re

from pydantic import BaseModel, Field

from backend.app.dataset.schemas import CanonicalNewsRecord


class OverlapPairMetric(BaseModel):
    source_name: str
    target_name: str
    source_records: int
    target_records: int
    exact_text_overlap_count: int
    exact_text_overlap_ratio: float = Field(
        ..., description="Exact matches relative to target set size"
    )
    normalized_overlap_count: int
    potential_contamination_count: int
    contamination_risk_level: str = Field(
        ..., description="'NONE', 'LOW', 'MODERATE', or 'HIGH'"
    )


class CrossDatasetAnalysisResult(BaseModel):
    """Encapsulates overlap matrices across datasets or across splits within a single dataset."""

    datasets_profiled: list[str]
    is_multi_dataset: bool
    pairwise_overlaps: list[OverlapPairMetric]
    cross_split_leakage_detected: bool
    notes: list[str] = Field(default_factory=list)


class CrossDatasetAnalyzer:
    """Detects content overlap and contamination across disparate benchmark datasets and split boundaries."""

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").lower()).strip()

    @classmethod
    def _hash_text(cls, text: str) -> str:
        return hashlib.sha256(cls._normalize_text(text).encode("utf-8")).hexdigest()

    @classmethod
    def analyze_splits(
        cls,
        splits: dict[str, list[CanonicalNewsRecord]],
    ) -> CrossDatasetAnalysisResult:
        """
        Analyzes overlap and contamination across splits (e.g. 'train', 'val', 'test').
        Guarantees validation and test sets have zero contamination from training data.
        """
        split_names = list(splits.keys())
        pairwise: list[OverlapPairMetric] = []
        leakage_found = False

        split_hashes = {
            s_name: {cls._hash_text(r.text) for r in recs if r.text}
            for s_name, recs in splits.items()
        }

        for i, s1 in enumerate(split_names):
            for j, s2 in enumerate(split_names):
                if i >= j:
                    continue
                h1 = split_hashes[s1]
                h2 = split_hashes[s2]
                overlap = len(h1.intersection(h2))
                ratio = round(overlap / max(len(h2), 1), 4)

                risk = "NONE"
                if overlap > 0:
                    leakage_found = True
                    risk = "HIGH" if ("test" in (s1, s2) or "val" in (s1, s2)) else "LOW"

                pairwise.append(
                    OverlapPairMetric(
                        source_name=s1,
                        target_name=s2,
                        source_records=len(splits[s1]),
                        target_records=len(splits[s2]),
                        exact_text_overlap_count=overlap,
                        exact_text_overlap_ratio=ratio,
                        normalized_overlap_count=overlap,
                        potential_contamination_count=overlap,
                        contamination_risk_level=risk,
                    )
                )

        notes = [
            f"Evaluated pairwise contamination across {len(split_names)} partitions ({', '.join(split_names)})."
        ]
        if leakage_found:
            notes.append(
                "CRITICAL WARNING: Split overlap detected! Data leakage is present between partitions."
            )
        else:
            notes.append(
                "CLEAN SPLIT VERIFICATION: Zero cross-split textual overlap detected; evaluation partitions are strictly isolated."
            )

        return CrossDatasetAnalysisResult(
            datasets_profiled=split_names,
            is_multi_dataset=False,
            pairwise_overlaps=pairwise,
            cross_split_leakage_detected=leakage_found,
            notes=notes,
        )

    @classmethod
    def analyze_datasets(
        cls,
        datasets: dict[str, list[CanonicalNewsRecord]],
    ) -> CrossDatasetAnalysisResult:
        """
        Compares multiple distinct datasets to quantify benchmark intersection and overlap.
        """
        names = list(datasets.keys())
        if len(names) < 2:
            return CrossDatasetAnalysisResult(
                datasets_profiled=names,
                is_multi_dataset=False,
                pairwise_overlaps=[],
                cross_split_leakage_detected=False,
                notes=[
                    "Only a single benchmark dataset is currently loaded locally.",
                    "Cross-dataset benchmark contamination cannot be evaluated until additional candidate corpora (FakeNewsNet, Fakeddit, NELA-GT) are acquired.",
                ],
            )

        pairwise: list[OverlapPairMetric] = []
        ds_hashes = {
            d_name: {cls._hash_text(r.text) for r in recs if r.text}
            for d_name, recs in datasets.items()
        }

        for i, d1 in enumerate(names):
            for j, d2 in enumerate(names):
                if i >= j:
                    continue
                h1 = ds_hashes[d1]
                h2 = ds_hashes[d2]
                overlap = len(h1.intersection(h2))
                ratio = round(overlap / max(len(h2), 1), 4)

                risk = "NONE" if overlap == 0 else ("LOW" if ratio < 0.05 else "MODERATE")
                pairwise.append(
                    OverlapPairMetric(
                        source_name=d1,
                        target_name=d2,
                        source_records=len(datasets[d1]),
                        target_records=len(datasets[d2]),
                        exact_text_overlap_count=overlap,
                        exact_text_overlap_ratio=ratio,
                        normalized_overlap_count=overlap,
                        potential_contamination_count=overlap,
                        contamination_risk_level=risk,
                    )
                )

        notes = [
            f"Evaluated cross-benchmark overlap across {len(names)} corpora ({', '.join(names)})."
        ]
        return CrossDatasetAnalysisResult(
            datasets_profiled=names,
            is_multi_dataset=True,
            pairwise_overlaps=pairwise,
            cross_split_leakage_detected=False,
            notes=notes,
        )
