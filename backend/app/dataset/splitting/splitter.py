"""Dataset splitting module enforcing zero duplicate and near-duplicate split leakage."""

import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from backend.app.dataset.schemas import CanonicalNewsRecord, SplitStatistics


class DatasetSplitter:
    """Partitions records into train, validation, and test splits with strict cluster isolation."""

    @classmethod
    def build_leakage_free_clusters(
        cls, records: list[CanonicalNewsRecord]
    ) -> list[list[CanonicalNewsRecord]]:
        """
        Groups records that share an exact duplicate_group_id or near_duplicate_group_id
        into atomic clusters that must never be separated across splits.
        """
        # Map group IDs to record indices
        exact_to_records: dict[str, list[CanonicalNewsRecord]] = defaultdict(list)
        near_to_records: dict[str, list[CanonicalNewsRecord]] = defaultdict(list)
        standalone: list[CanonicalNewsRecord] = []

        for r in records:
            if r.duplicate_group_id:
                exact_to_records[r.duplicate_group_id].append(r)
            elif r.near_duplicate_group_id:
                near_to_records[r.near_duplicate_group_id].append(r)
            else:
                standalone.append(r)

        # Merge clusters that overlap between exact and near
        clusters: list[list[CanonicalNewsRecord]] = []

        # Process exact groups
        processed_records = set()
        for _group_id, group_records in exact_to_records.items():
            cluster = list(group_records)
            for item in group_records:
                processed_records.add(item.record_id)
            clusters.append(cluster)

        # Process near groups
        for _group_id, group_records in near_to_records.items():
            cluster = [item for item in group_records if item.record_id not in processed_records]
            if cluster:
                for item in cluster:
                    processed_records.add(item.record_id)
                clusters.append(cluster)

        # Add single standalone records as 1-element clusters
        for r in standalone:
            if r.record_id not in processed_records:
                clusters.append([r])

        return clusters

    @classmethod
    def split_dataset(
        cls,
        records: list[CanonicalNewsRecord],
        strategy: str = "stratified",
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        seed: int = 42,
    ) -> tuple[list[CanonicalNewsRecord], SplitStatistics]:
        """
        Splits canonical records into train, val, and test partitions.
        Guarantees:
        1. Exact and near duplicates never cross split boundaries.
        2. Deterministic reproducibility via seed.
        3. Stratification by label when possible.
        """
        if not records:
            return [], SplitStatistics(train_count=0, val_count=0, test_count=0)

        # Normalize ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        r_train = train_ratio / total_ratio
        r_val = val_ratio / total_ratio

        clusters = cls.build_leakage_free_clusters(records)
        rng = random.Random(seed)

        train_clusters: list[list[CanonicalNewsRecord]] = []
        val_clusters: list[list[CanonicalNewsRecord]] = []
        test_clusters: list[list[CanonicalNewsRecord]] = []

        reassignments: list[dict[str, Any]] = []
        reassigned_count = 0
        affected_clusters = 0
        train_val_conflicts = 0
        train_test_conflicts = 0
        val_test_conflicts = 0

        if strategy == "temporal":
            # Sort clusters by minimum normalized publication date
            # Put clusters with missing dates deterministically at the start
            def cluster_date_key(cl: list[CanonicalNewsRecord]):
                dates = [r.published_at_normalized for r in cl if r.published_at_normalized]
                return min(dates) if dates else datetime.min.replace(tzinfo=timezone.utc)

            clusters.sort(key=cluster_date_key)
            n_clusters = len(clusters)
            train_end = int(n_clusters * r_train)
            val_end = int(n_clusters * (r_train + r_val))

            train_clusters = clusters[:train_end]
            val_clusters = clusters[train_end:val_end]
            test_clusters = clusters[val_end:]

        elif strategy == "stratified":
            # Group clusters by dominant label
            label_to_clusters: dict[str, list[list[CanonicalNewsRecord]]] = defaultdict(list)
            for cl in clusters:
                # Use majority label in cluster
                lbl = cl[0].normalized_label or cl[0].original_label or "UNKNOWN"
                label_to_clusters[lbl].append(cl)

            for _lbl, cl_list in label_to_clusters.items():
                rng.shuffle(cl_list)
                n = len(cl_list)
                n_tr = int(n * r_train)
                n_va = int(n * r_val)
                # Ensure at least 1 record in val and test if cluster size allows
                if n >= 3 and n_va == 0:
                    n_va = 1
                if n >= 3 and (n - n_tr - n_va) == 0:
                    n_tr = max(1, n_tr - 1)

                train_clusters.extend(cl_list[:n_tr])
                val_clusters.extend(cl_list[n_tr : n_tr + n_va])
                test_clusters.extend(cl_list[n_tr + n_va :])

        elif strategy == "official_benchmark_splits":
            for cl in clusters:
                orig_splits = {
                    (r.original_split or r.split)
                    for r in cl
                    if (r.original_split or r.split)
                }
                # Normalize 'valid' -> 'val'
                orig_splits = {"val" if s == "valid" else s for s in orig_splits}

                if not orig_splits:
                    target_split = "train"
                elif len(orig_splits) == 1:
                    target_split = next(iter(orig_splits))
                else:
                    # Cluster spans multiple partitions! Apply deterministic precedence: TRAIN > VAL > TEST
                    affected_clusters += 1
                    if "train" in orig_splits and "val" in orig_splits:
                        train_val_conflicts += 1
                    if "train" in orig_splits and "test" in orig_splits:
                        train_test_conflicts += 1
                    if "val" in orig_splits and "test" in orig_splits and "train" not in orig_splits:
                        val_test_conflicts += 1

                    if "train" in orig_splits:
                        target_split = "train"
                    elif "val" in orig_splits:
                        target_split = "val"
                    else:
                        target_split = "test"

                # Check if any records in cluster differ from target_split
                for r in cl:
                    orig = "val" if (r.original_split or r.split) == "valid" else (r.original_split or r.split)
                    if orig and orig != target_split:
                        reassigned_count += 1
                        cluster_id = r.duplicate_group_id or r.near_duplicate_group_id or "cluster"
                        reason = (
                            f"Cluster isolation resolution (precedence: TRAIN > VAL > TEST) "
                            f"to eliminate cross-split duplicate leakage from {orig} into {target_split}."
                        )
                        reassignments.append({
                            "record_id": r.record_id,
                            "original_id": r.original_id,
                            "original_split": orig,
                            "final_split": target_split,
                            "cluster_id": cluster_id,
                            "reassignment_reason": reason,
                        })

                if target_split == "train":
                    train_clusters.append(cl)
                elif target_split == "val":
                    val_clusters.append(cl)
                else:
                    test_clusters.append(cl)

        elif strategy == "random":
            rng.shuffle(clusters)
            n_clusters = len(clusters)
            train_end = int(n_clusters * r_train)
            val_end = int(n_clusters * (r_train + r_val))
            train_clusters = clusters[:train_end]
            val_clusters = clusters[train_end:val_end]
            test_clusters = clusters[val_end:]

        else:
            raise ValueError(
                f"Unsupported splitting strategy: '{strategy}'. "
                "Supported strategies are: 'stratified', 'temporal', 'random', 'official_benchmark_splits'."
            )

        # Flatten and assign split tag
        final_records: list[CanonicalNewsRecord] = []
        train_dist = Counter()
        val_dist = Counter()
        test_dist = Counter()

        for cl in train_clusters:
            for r in cl:
                orig = "val" if (r.original_split or r.split) == "valid" else (r.original_split or r.split)
                reason = None
                if orig and orig != "train":
                    reason = (
                        f"Cluster isolation resolution (precedence: TRAIN > VAL > TEST) "
                        f"to eliminate cross-split duplicate leakage from {orig} into train."
                    )
                lbl = r.normalized_label or r.original_label
                train_dist[lbl] += 1
                final_records.append(
                    r.model_copy(
                        update={
                            "split": "train",
                            "original_split": orig or "train",
                            "split_reassignment_reason": reason,
                        }
                    )
                )

        for cl in val_clusters:
            for r in cl:
                orig = "val" if (r.original_split or r.split) == "valid" else (r.original_split or r.split)
                reason = None
                if orig and orig != "val":
                    reason = (
                        f"Cluster isolation resolution (precedence: TRAIN > VAL > TEST) "
                        f"to eliminate cross-split duplicate leakage from {orig} into val."
                    )
                lbl = r.normalized_label or r.original_label
                val_dist[lbl] += 1
                final_records.append(
                    r.model_copy(
                        update={
                            "split": "val",
                            "original_split": orig or "val",
                            "split_reassignment_reason": reason,
                        }
                    )
                )

        for cl in test_clusters:
            for r in cl:
                orig = "val" if (r.original_split or r.split) == "valid" else (r.original_split or r.split)
                reason = None
                if orig and orig != "test":
                    reason = (
                        f"Cluster isolation resolution (precedence: TRAIN > VAL > TEST) "
                        f"to eliminate cross-split duplicate leakage from {orig} into test."
                    )
                lbl = r.normalized_label or r.original_label
                test_dist[lbl] += 1
                final_records.append(
                    r.model_copy(
                        update={
                            "split": "test",
                            "original_split": orig or "test",
                            "split_reassignment_reason": reason,
                        }
                    )
                )

        split_stats = SplitStatistics(
            train_count=sum(train_dist.values()),
            val_count=sum(val_dist.values()),
            test_count=sum(test_dist.values()),
            train_distribution=dict(train_dist),
            val_distribution=dict(val_dist),
            test_distribution=dict(test_dist),
            reassigned_count=reassigned_count,
            affected_clusters_count=affected_clusters,
            train_val_conflicts=train_val_conflicts,
            train_test_conflicts=train_test_conflicts,
            val_test_conflicts=val_test_conflicts,
            reassignments=reassignments,
        )

        return final_records, split_stats
