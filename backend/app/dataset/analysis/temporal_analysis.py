"""Temporal coverage, chronological distribution, and label stability analysis module."""

from collections import Counter, defaultdict
from datetime import datetime

from pydantic import BaseModel, Field

from backend.app.dataset.schemas import CanonicalNewsRecord


class TemporalAnalysisResult(BaseModel):
    """Encapsulates timestamp completeness, temporal span, and longitudinal class stability."""

    total_records: int
    timestamps_available: int
    timestamps_missing: int
    timestamp_coverage_ratio: float
    has_temporal_data: bool
    earliest_timestamp: str | None = None
    latest_timestamp: str | None = None
    records_per_year: dict[str, int] = Field(default_factory=dict)
    records_per_month: dict[str, int] = Field(default_factory=dict)
    yearly_label_distribution: dict[str, dict[str, int]] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class TemporalAnalyzer:
    """Evaluates timestamp availability, chronological span, and temporal class distribution."""

    @classmethod
    def analyze(cls, records: list[CanonicalNewsRecord]) -> TemporalAnalysisResult:
        if not records:
            return TemporalAnalysisResult(
                total_records=0,
                timestamps_available=0,
                timestamps_missing=0,
                timestamp_coverage_ratio=0.0,
                has_temporal_data=False,
                notes=["Empty dataset provided."],
            )

        total = len(records)
        parsed_dates: list[datetime] = []
        year_counter: Counter[str] = Counter()
        month_counter: Counter[str] = Counter()
        year_label_map: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for r in records:
            dt = r.published_at_normalized
            # If not normalized, attempt parsing raw if present
            if dt is None and r.published_at_raw:
                try:
                    dt = datetime.fromisoformat(r.published_at_raw.replace("Z", "+00:00"))
                except Exception:
                    dt = None

            if dt is not None:
                parsed_dates.append(dt)
                yr_str = str(dt.year)
                mo_str = dt.strftime("%Y-%m")
                lbl = r.normalized_label or "UNRESOLVED"

                year_counter[yr_str] += 1
                month_counter[mo_str] += 1
                year_label_map[yr_str][lbl] += 1

        avail = len(parsed_dates)
        missing = total - avail
        coverage = round(avail / total, 4)

        if not parsed_dates:
            return TemporalAnalysisResult(
                total_records=total,
                timestamps_available=0,
                timestamps_missing=total,
                timestamp_coverage_ratio=0.0,
                has_temporal_data=False,
                notes=[
                    "No valid publication timestamps detected in this dataset.",
                    "Temporal distribution and drift analysis cannot be empirically evaluated.",
                    "Temporal splits should not be applied to datasets lacking validated chronological metadata.",
                ],
            )

        parsed_dates.sort()
        earliest = parsed_dates[0].isoformat()
        latest = parsed_dates[-1].isoformat()

        notes = [
            f"Temporal coverage spans from {earliest[:10]} to {latest[:10]} across {avail} records ({round(coverage * 100, 2)}% coverage).",
            f"Chronological activity spans {len(year_counter)} distinct calendar years.",
        ]
        if coverage < 0.50:
            notes.append(
                "High rate of missing publication timestamps (>50%); chronological analysis must be interpreted with extreme caution."
            )

        return TemporalAnalysisResult(
            total_records=total,
            timestamps_available=avail,
            timestamps_missing=missing,
            timestamp_coverage_ratio=coverage,
            has_temporal_data=True,
            earliest_timestamp=earliest,
            latest_timestamp=latest,
            records_per_year=dict(sorted(year_counter.items())),
            records_per_month=dict(sorted(month_counter.items())),
            yearly_label_distribution={yr: dict(dist) for yr, dist in sorted(year_label_map.items())},
            notes=notes,
        )
