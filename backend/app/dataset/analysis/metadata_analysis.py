"""Metadata profiling, attribute missingness, and categorical distributions module."""

import math
from collections import Counter
from typing import Any

from pydantic import BaseModel, Field

from backend.app.dataset.schemas import CanonicalNewsRecord


class MetadataFieldSummary(BaseModel):
    field_name: str
    available_count: int
    missing_count: int
    missing_percentage: float
    unique_values: int
    normalized_entropy: float = Field(
        ..., description="Normalized Shannon entropy measuring category dispersion"
    )
    top_values: list[dict[str, Any]] = Field(default_factory=list)


class MetadataAnalysisResult(BaseModel):
    """Encapsulates completeness, cardinality, and entropy distributions for all record metadata."""

    total_records: int
    metadata_fields_profiled: list[str]
    field_summaries: dict[str, MetadataFieldSummary]
    overall_completeness_ratio: float = Field(
        ..., description="Mean availability across all tracked metadata fields"
    )
    notes: list[str] = Field(default_factory=list)


class MetadataAnalyzer:
    """Extracts and evaluates attribute missingness, uniqueness, and categorical distribution."""

    TRACKED_FIELDS = [
        "author",
        "title",
        "source_name_raw",
        "source_domain",
        "speaker",
        "party_affiliation",
        "context",
        "state_info",
        "subjects",
        "job_title",
    ]

    @classmethod
    def _compute_field_summary(
        cls, field_name: str, values: list[Any], total_records: int
    ) -> MetadataFieldSummary:
        non_null_str = [
            str(v).strip()
            for v in values
            if v is not None and str(v).strip() not in ("", "none", "nan", "null")
        ]
        avail = len(non_null_str)
        missing = total_records - avail
        missing_pct = round((missing / total_records) * 100, 2) if total_records > 0 else 100.0

        counter = Counter(non_null_str)
        unique_cnt = len(counter)

        # Top 10 values
        top_vals = [
            {"value": val, "count": cnt, "percentage": round((cnt / total_records) * 100, 2)}
            for val, cnt in counter.most_common(10)
        ]

        # Normalized Shannon entropy
        if unique_cnt > 1 and avail > 0:
            h = -sum(
                (cnt / avail) * math.log2(cnt / avail) for cnt in counter.values() if cnt > 0
            )
            norm_entropy = round(h / math.log2(unique_cnt), 4)
        else:
            norm_entropy = 0.0

        return MetadataFieldSummary(
            field_name=field_name,
            available_count=avail,
            missing_count=missing,
            missing_percentage=missing_pct,
            unique_values=unique_cnt,
            normalized_entropy=norm_entropy,
            top_values=top_vals,
        )

    @classmethod
    def analyze(cls, records: list[CanonicalNewsRecord]) -> MetadataAnalysisResult:
        if not records:
            return MetadataAnalysisResult(
                total_records=0,
                metadata_fields_profiled=[],
                field_summaries={},
                overall_completeness_ratio=0.0,
                notes=["Empty records dataset provided."],
            )

        total = len(records)
        extracted: dict[str, list[Any]] = {f: [] for f in cls.TRACKED_FIELDS}

        # Check for any extra keys in metadata dict
        extra_keys: set[str] = set()
        for r in records:
            # First-class fields
            extracted["author"].append(r.author)
            extracted["title"].append(r.title)
            extracted["source_name_raw"].append(r.source_name_raw)
            extracted["source_domain"].append(r.source_domain)

            # Metadata sub-dictionary fields
            meta = r.metadata or {}
            for k in ["speaker", "party_affiliation", "context", "state_info", "subjects", "job_title"]:
                extracted[k].append(meta.get(k))

            for ek in meta.keys():
                if ek not in cls.TRACKED_FIELDS and ek != "credit_history":
                    extra_keys.add(ek)

        # Profile discovered extra fields if present across records
        for ek in sorted(extra_keys):
            extracted[ek] = [r.metadata.get(ek) for r in records]

        summaries: dict[str, MetadataFieldSummary] = {}
        avail_ratios: list[float] = []

        for field_name, vals in extracted.items():
            summary = cls._compute_field_summary(field_name, vals, total)
            # Only keep field if at least one record has a non-null entry or if it is a core field
            if summary.available_count > 0 or field_name in ("speaker", "party_affiliation", "author"):
                summaries[field_name] = summary
                avail_ratios.append(summary.available_count / total)

        overall_completeness = round(sum(avail_ratios) / len(avail_ratios), 4) if avail_ratios else 0.0

        notes = [
            f"Profiled {len(summaries)} distinct metadata fields across {total} records.",
            f"Mean field-level attribute availability is {round(overall_completeness * 100, 2)}%.",
        ]

        return MetadataAnalysisResult(
            total_records=total,
            metadata_fields_profiled=list(summaries.keys()),
            field_summaries=summaries,
            overall_completeness_ratio=overall_completeness,
            notes=notes,
        )
