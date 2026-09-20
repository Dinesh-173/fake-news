"""Deduplication module performing deterministic exact and near-duplicate detection."""

import hashlib
import re
from collections import defaultdict

from backend.app.dataset.schemas import CanonicalNewsRecord, DuplicateStatistics


class Deduplicator:
    """Detects exact and near-duplicates and assigns cluster IDs for split isolation."""

    @classmethod
    def compute_content_hash(cls, text: str) -> str:
        """Computes deterministic SHA-256 hash over lowercased, collapsed text."""
        norm = re.sub(r"\s+", " ", (text or "").lower()).strip()
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    @classmethod
    def extract_ngrams(cls, text: str, n: int = 5) -> set[str]:
        """Extracts character n-grams for Jaccard similarity estimation."""
        norm = re.sub(r"\s+", " ", (text or "").lower()).strip()
        if len(norm) < n:
            return {norm}
        return {norm[i : i + n] for i in range(len(norm) - n + 1)}

    @classmethod
    def compute_jaccard_similarity(cls, set_a: set[str], set_b: set[str]) -> float:
        """Computes Jaccard similarity between two sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0

    @classmethod
    def process_duplicates(
        cls,
        records: list[CanonicalNewsRecord],
        near_duplicate_threshold: float = 0.85,
        max_near_duplicate_comparisons: int = 5000,
    ) -> tuple[list[CanonicalNewsRecord], DuplicateStatistics]:
        """
        Executes exact and near-duplicate clustering.
        Records in the same duplicate or near-duplicate group receive group IDs
        that the splitter will use to guarantee zero cross-split leakage.
        """
        # --- Stage 1: Exact Deduplication ---
        hash_to_records: dict[str, list[CanonicalNewsRecord]] = defaultdict(list)
        for r in records:
            content_hash = cls.compute_content_hash(r.text)
            r_updated = r.model_copy(update={"content_hash": content_hash})
            hash_to_records[content_hash].append(r_updated)

        exact_dups_count = 0
        exact_dup_groups_count = 0
        deduped_records: list[CanonicalNewsRecord] = []

        for content_hash, cluster in hash_to_records.items():
            if len(cluster) == 1:
                deduped_records.append(cluster[0])
            else:
                exact_dup_groups_count += 1
                group_id = f"exact_grp_{content_hash[:12]}"
                # Deterministic selection of primary record: highest metadata count, else first
                primary = max(
                    cluster,
                    key=lambda x: (
                        len(x.metadata or {}),
                        bool(x.title),
                        bool(x.published_at_normalized),
                    ),
                )
                for item in cluster:
                    if item.record_id == primary.record_id:
                        deduped_records.append(
                            item.model_copy(
                                update={
                                    "duplicate_group_id": group_id,
                                    "is_duplicate": False,
                                    "duplicate_of": None,
                                }
                            )
                        )
                    else:
                        exact_dups_count += 1
                        deduped_records.append(
                            item.model_copy(
                                update={
                                    "duplicate_group_id": group_id,
                                    "is_duplicate": True,
                                    "duplicate_of": primary.record_id,
                                }
                            )
                        )

        # --- Stage 2: Near-Duplicate Detection (Conservative N-Gram Jaccard) ---
        # Only compare distinct non-exact-duplicate primary records to keep performance bounded
        near_dup_groups: dict[str, str] = {}  # record_id -> near_dup_group_id
        near_dup_clusters: list[list[str]] = []

        primary_records = [r for r in deduped_records if not r.is_duplicate]
        sample_size = min(len(primary_records), max_near_duplicate_comparisons)

        if sample_size > 1:
            ngram_cache = [cls.extract_ngrams(r.text) for r in primary_records[:sample_size]]
            visited: set[int] = set()

            for i in range(sample_size):
                if i in visited:
                    continue
                current_cluster = [primary_records[i].record_id]
                for j in range(i + 1, sample_size):
                    if j in visited:
                        continue
                    sim = cls.compute_jaccard_similarity(ngram_cache[i], ngram_cache[j])
                    if sim >= near_duplicate_threshold:
                        visited.add(j)
                        current_cluster.append(primary_records[j].record_id)

                if len(current_cluster) > 1:
                    visited.add(i)
                    near_dup_clusters.append(current_cluster)
                    group_id = f"near_grp_{current_cluster[0][:8]}"
                    for rec_id in current_cluster:
                        near_dup_groups[rec_id] = group_id

        # Update records with near_duplicate_group_id
        final_records: list[CanonicalNewsRecord] = []
        for r in deduped_records:
            if r.record_id in near_dup_groups:
                final_records.append(
                    r.model_copy(update={"near_duplicate_group_id": near_dup_groups[r.record_id]})
                )
            else:
                final_records.append(r)

        stats = DuplicateStatistics(
            total_records=len(records),
            unique_content_hashes=len(hash_to_records),
            exact_duplicates_count=exact_dups_count,
            exact_duplicate_groups=exact_dup_groups_count,
            near_duplicate_groups=len(near_dup_clusters),
        )

        return final_records, stats
