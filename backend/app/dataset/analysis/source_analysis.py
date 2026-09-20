"""Source distribution, publisher domain analysis, and concentration metrics module."""

from collections import Counter, defaultdict

from pydantic import BaseModel, Field

from backend.app.dataset.schemas import CanonicalNewsRecord


class SourceFrequencyItem(BaseModel):
    name: str
    count: int
    percentage: float


class SourceAnalysisResult(BaseModel):
    """Encapsulates empirical source distribution and concentration metrics."""

    total_records: int
    unique_sources: int
    unique_domains: int
    top_publishers: list[SourceFrequencyItem]
    top_domains: list[SourceFrequencyItem]
    top_5_share: float = Field(..., description="Percentage of total records from top 5 sources")
    top_10_share: float = Field(..., description="Percentage of total records from top 10 sources")
    herfindahl_index: float = Field(
        ...,
        description="Herfindahl-Hirschman Index (HHI) of source concentration in range [0, 10000]",
    )
    normalized_hhi: float = Field(
        ...,
        description="HHI normalized between 0 (perfect dispersion) and 1 (single monopolistic source)",
    )
    source_label_contingency: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Cross-tabulation of top 10 sources against normalized veracity labels",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Descriptive notes contextualizing observed distribution properties",
    )


class SourceAnalyzer:
    """Measures source diversity, domain distribution, and market concentration."""

    @classmethod
    def analyze(
        cls,
        records: list[CanonicalNewsRecord],
        top_k: int = 15,
    ) -> SourceAnalysisResult:
        if not records:
            return SourceAnalysisResult(
                total_records=0,
                unique_sources=0,
                unique_domains=0,
                top_publishers=[],
                top_domains=[],
                top_5_share=0.0,
                top_10_share=0.0,
                herfindahl_index=0.0,
                normalized_hhi=0.0,
                source_label_contingency={},
                notes=["Empty record set provided; no source metrics computed."],
            )

        total = len(records)
        source_counter: Counter[str] = Counter()
        domain_counter: Counter[str] = Counter()
        source_label_map: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for r in records:
            src = r.source_name_normalized or r.author or r.source_name_raw or "UNKNOWN_SOURCE"
            dom = r.source_domain or "UNKNOWN_DOMAIN"
            norm_label = r.normalized_label or "UNRESOLVED"

            source_counter[src] += 1
            domain_counter[dom] += 1
            source_label_map[src][norm_label] += 1

        # Sorted lists
        sorted_sources = source_counter.most_common()
        sorted_domains = domain_counter.most_common()

        top_publishers = [
            SourceFrequencyItem(
                name=name, count=cnt, percentage=round((cnt / total) * 100, 2)
            )
            for name, cnt in sorted_sources[:top_k]
        ]
        top_domains = [
            SourceFrequencyItem(
                name=dom, count=cnt, percentage=round((cnt / total) * 100, 2)
            )
            for dom, cnt in sorted_domains[:top_k]
        ]

        # Concentration shares
        top_5_count = sum(cnt for _, cnt in sorted_sources[:5])
        top_10_count = sum(cnt for _, cnt in sorted_sources[:10])
        top_5_share = round((top_5_count / total) * 100, 2)
        top_10_share = round((top_10_count / total) * 100, 2)

        # Herfindahl-Hirschman Index: sum of (market_share_percentage)^2
        # Values range from close to 0 (highly atomized) to 10,000 (monopoly)
        hhi = sum(((cnt / total) * 100) ** 2 for _, cnt in sorted_sources)
        hhi = round(hhi, 2)

        # Normalized HHI: H* = (H - 1/N) / (1 - 1/N) where N is number of unique sources
        n_sources = len(sorted_sources)
        if n_sources > 1:
            h_ratio = hhi / 10000.0
            norm_hhi = (h_ratio - (1.0 / n_sources)) / (1.0 - (1.0 / n_sources))
            norm_hhi = round(max(0.0, min(1.0, norm_hhi)), 4)
        else:
            norm_hhi = 1.0

        # Build contingency matrix for top 10 sources
        contingency: dict[str, dict[str, int]] = {}
        for src, _ in sorted_sources[:10]:
            contingency[src] = dict(source_label_map[src])

        notes: list[str] = [
            f"Dataset encompasses {n_sources} distinct sources/speakers and {len(sorted_domains)} domain entries.",
            f"Top-5 sources account for {top_5_share}% of all records; top-10 account for {top_10_share}%.",
            f"Observed Herfindahl-Hirschman Index is {hhi} (Normalized: {norm_hhi}).",
        ]
        if norm_hhi > 0.15:
            notes.append(
                "Source concentration is moderate-to-high; empirical analysis indicates a subset of prominent speakers or publishers constitutes a large proportion of statements."
            )
        else:
            notes.append(
                "Source distribution shows broad dispersion across a wide diversity of originators."
            )

        return SourceAnalysisResult(
            total_records=total,
            unique_sources=n_sources,
            unique_domains=len(sorted_domains),
            top_publishers=top_publishers,
            top_domains=top_domains,
            top_5_share=top_5_share,
            top_10_share=top_10_share,
            herfindahl_index=hhi,
            normalized_hhi=norm_hhi,
            source_label_contingency=contingency,
            notes=notes,
        )
