"""Duplicate identification, cluster size distribution, and label divergence analysis module."""

from collections import defaultdict

from pydantic import BaseModel, Field

from backend.app.dataset.schemas import CanonicalNewsRecord


class ConflictingClusterReport(BaseModel):
    cluster_id: str
    cluster_type: str = Field(..., description="'exact' or 'near'")
    record_count: int
    normalized_labels: list[str]
    sample_text: str


class ClusterSizeDistribution(BaseModel):
    singletons: int = 0
    size_2: int = 0
    size_3_to_5: int = 0
    size_gt_5: int = 0
    max_cluster_size: int = 1


class DuplicateAnalysisResult(BaseModel):
    """Encapsulates exact and near-duplicate counts, cluster dynamics, and veracity conflicts."""

    total_records: int
    exact_duplicates_count: int
    exact_duplicates_percentage: float
    exact_duplicate_groups: int
    near_duplicate_groups: int
    near_duplicate_records_count: int
    near_duplicate_percentage: float
    cluster_size_distribution: ClusterSizeDistribution
    conflicting_clusters_count: int = Field(
        ..., description="Number of duplicate clusters exhibiting contradictory normalized labels"
    )
    conflicting_clusters: list[ConflictingClusterReport] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class DuplicateAnalyzer:
    """Profiles exact and near-duplicate structures and audits veracity divergence across duplicate clusters."""

    @classmethod
    def analyze(cls, records: list[CanonicalNewsRecord]) -> DuplicateAnalysisResult:
        if not records:
            return DuplicateAnalysisResult(
                total_records=0,
                exact_duplicates_count=0,
                exact_duplicates_percentage=0.0,
                exact_duplicate_groups=0,
                near_duplicate_groups=0,
                near_duplicate_records_count=0,
                near_duplicate_percentage=0.0,
                cluster_size_distribution=ClusterSizeDistribution(),
                conflicting_clusters_count=0,
                conflicting_clusters=[],
                notes=["Empty dataset provided; no duplicate metrics computed."],
            )

        total = len(records)
        exact_dups = sum(1 for r in records if r.is_duplicate)
        exact_pct = round((exact_dups / total) * 100, 2)

        # Exact clusters
        exact_cluster_map: dict[str, list[CanonicalNewsRecord]] = defaultdict(list)
        for r in records:
            grp = r.duplicate_group_id or r.content_hash
            if grp:
                exact_cluster_map[grp].append(r)

        exact_groups_count = sum(1 for cl in exact_cluster_map.values() if len(cl) > 1)

        # Near clusters
        near_cluster_map: dict[str, list[CanonicalNewsRecord]] = defaultdict(list)
        near_rec_count = 0
        for r in records:
            if r.near_duplicate_group_id:
                near_cluster_map[r.near_duplicate_group_id].append(r)
                near_rec_count += 1

        near_pct = round((near_rec_count / total) * 100, 2)
        near_groups_count = len(near_cluster_map)

        # Cluster size aggregation (combining effective cluster memberships)
        all_cluster_sizes: list[int] = []
        for cl in exact_cluster_map.values():
            if len(cl) > 1:
                all_cluster_sizes.append(len(cl))

        dist = ClusterSizeDistribution()
        for sz in all_cluster_sizes:
            if sz == 2:
                dist.size_2 += 1
            elif 3 <= sz <= 5:
                dist.size_3_to_5 += 1
            elif sz > 5:
                dist.size_gt_5 += 1

        dist.max_cluster_size = max(all_cluster_sizes) if all_cluster_sizes else 1
        dist.singletons = total - (exact_dups + exact_groups_count)

        # Audit label divergence / conflict in clusters
        conflicts: list[ConflictingClusterReport] = []

        # 1. Check exact clusters for conflicting labels
        for grp_id, members in exact_cluster_map.items():
            if len(members) > 1:
                labels = {m.normalized_label for m in members if m.normalized_label}
                if len(labels) > 1:
                    conflicts.append(
                        ConflictingClusterReport(
                            cluster_id=grp_id,
                            cluster_type="exact",
                            record_count=len(members),
                            normalized_labels=sorted(labels),
                            sample_text=members[0].text[:120],
                        )
                    )

        # 2. Check near clusters for conflicting labels
        for grp_id, members in near_cluster_map.items():
            labels = {m.normalized_label for m in members if m.normalized_label}
            if len(labels) > 1:
                # Avoid duplicate conflict reports if already flagged in exact
                if not any(c.cluster_id == grp_id for c in conflicts):
                    conflicts.append(
                        ConflictingClusterReport(
                            cluster_id=grp_id,
                            cluster_type="near",
                            record_count=len(members),
                            normalized_labels=sorted(labels),
                            sample_text=members[0].text[:120],
                        )
                    )

        notes = [
            f"Exact duplicates comprise {exact_dups} records ({exact_pct}%) across {exact_groups_count} clusters.",
            f"Near-duplicate grouping identified {near_groups_count} clusters encompassing {near_rec_count} records ({near_pct}%).",
        ]
        if conflicts:
            notes.append(
                f"AUDIT WARNING: {len(conflicts)} duplicate/near-duplicate clusters exhibit label divergence (conflicting fact-checking verdicts for identical or near-identical statements)."
            )
        else:
            notes.append("No label conflicts detected within identical or near-duplicate clusters.")

        return DuplicateAnalysisResult(
            total_records=total,
            exact_duplicates_count=exact_dups,
            exact_duplicates_percentage=exact_pct,
            exact_duplicate_groups=exact_groups_count,
            near_duplicate_groups=near_groups_count,
            near_duplicate_records_count=near_rec_count,
            near_duplicate_percentage=near_pct,
            cluster_size_distribution=dist,
            conflicting_clusters_count=len(conflicts),
            conflicting_clusters=conflicts[:20],  # Keep top 20 examples for reporting
            notes=notes,
        )
