"""Unit tests for dataset splitting and cluster-based leakage prevention."""

from backend.app.dataset.schemas import CanonicalNewsRecord
from backend.app.dataset.splitting.splitter import DatasetSplitter


def test_splitter_zero_duplicate_leakage():
    """Verify that records with the same duplicate_group_id are NEVER split across train/val/test."""
    records = []
    # Create 30 records, with 5 pairs of duplicates
    for i in range(20):
        records.append(
            CanonicalNewsRecord(
                record_id=f"rec_{i}",
                dataset_name="liar",
                original_id=str(i),
                text=f"Unique statement number {i}",
                original_label="true" if i % 2 == 0 else "false",
                normalized_label="LIKELY_TRUE" if i % 2 == 0 else "LIKELY_FALSE",
            )
        )

    # Add 5 duplicates paired with existing records
    for i in range(5):
        dup_grp = f"grp_{i}"
        records[i] = records[i].model_copy(update={"duplicate_group_id": dup_grp})
        records.append(
            CanonicalNewsRecord(
                record_id=f"dup_{i}",
                dataset_name="liar",
                original_id=f"copy_{i}",
                text=records[i].text,
                original_label=records[i].original_label,
                normalized_label=records[i].normalized_label,
                duplicate_group_id=dup_grp,
                is_duplicate=True,
                duplicate_of=records[i].record_id,
            )
        )

    split_records, stats = DatasetSplitter.split_dataset(
        records,
        strategy="stratified",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    # Check that duplicates are in the same split
    rec_by_id = {r.record_id: r for r in split_records}
    for i in range(5):
        orig_split = rec_by_id[f"rec_{i}"].split
        dup_split = rec_by_id[f"dup_{i}"].split
        assert orig_split == dup_split, (
            f"Leakage detected! Record rec_{i} in {orig_split} while duplicate dup_{i} in {dup_split}"
        )


def test_splitter_deterministic_seed():
    """Verify identical random seed produces identical split assignments."""
    records = [
        CanonicalNewsRecord(
            record_id=f"r_{i}",
            dataset_name="liar",
            original_id=str(i),
            text=f"Statement {i}",
            original_label="true",
        )
        for i in range(50)
    ]

    res1, _ = DatasetSplitter.split_dataset(records, seed=123)
    res2, _ = DatasetSplitter.split_dataset(records, seed=123)

    splits1 = [r.split for r in res1]
    splits2 = [r.split for r in res2]
    assert splits1 == splits2


def test_official_benchmark_splits_preservation():
    """Verify official splits are preserved when no clusters conflict."""
    records = [
        CanonicalNewsRecord(
            record_id=f"r_{i}",
            dataset_name="liar",
            original_id=str(i),
            text=f"Unique text {i}",
            original_label="true",
            original_split="train" if i < 10 else ("val" if i < 15 else "test"),
        )
        for i in range(20)
    ]

    split_records, stats = DatasetSplitter.split_dataset(
        records, strategy="official_benchmark_splits"
    )

    assert stats.train_count == 10
    assert stats.val_count == 5
    assert stats.test_count == 5
    assert stats.reassigned_count == 0
    assert stats.affected_clusters_count == 0


def test_official_benchmark_splits_conflict_resolution():
    """Verify deterministic conflict resolution (TRAIN > VAL > TEST) when clusters span splits."""
    records = [
        # Cluster 1: Train + Val -> must resolve to Train
        CanonicalNewsRecord(
            record_id="rec_1_tr",
            dataset_name="liar",
            original_id="101",
            text="Statement about economy",
            original_label="true",
            original_split="train",
            duplicate_group_id="grp_1",
        ),
        CanonicalNewsRecord(
            record_id="rec_1_va",
            dataset_name="liar",
            original_id="102",
            text="Statement about economy",
            original_label="true",
            original_split="val",
            duplicate_group_id="grp_1",
            is_duplicate=True,
            duplicate_of="rec_1_tr",
        ),
        # Cluster 2: Train + Test -> must resolve to Train
        CanonicalNewsRecord(
            record_id="rec_2_tr",
            dataset_name="liar",
            original_id="201",
            text="Statement about health",
            original_label="false",
            original_split="train",
            duplicate_group_id="grp_2",
        ),
        CanonicalNewsRecord(
            record_id="rec_2_te",
            dataset_name="liar",
            original_id="202",
            text="Statement about health",
            original_label="false",
            original_split="test",
            duplicate_group_id="grp_2",
            is_duplicate=True,
            duplicate_of="rec_2_tr",
        ),
        # Cluster 3: Val + Test -> must resolve to Val
        CanonicalNewsRecord(
            record_id="rec_3_va",
            dataset_name="liar",
            original_id="301",
            text="Statement about space",
            original_label="true",
            original_split="val",
            duplicate_group_id="grp_3",
        ),
        CanonicalNewsRecord(
            record_id="rec_3_te",
            dataset_name="liar",
            original_id="302",
            text="Statement about space",
            original_label="true",
            original_split="test",
            duplicate_group_id="grp_3",
            is_duplicate=True,
            duplicate_of="rec_3_va",
        ),
        # Singletons in test
        CanonicalNewsRecord(
            record_id="rec_clean_test",
            dataset_name="liar",
            original_id="401",
            text="Clean test statement",
            original_label="false",
            original_split="test",
        ),
    ]

    split_records, stats = DatasetSplitter.split_dataset(
        records, strategy="official_benchmark_splits"
    )

    rec_by_id = {r.record_id: r for r in split_records}

    # Cluster 1: both must be train
    assert rec_by_id["rec_1_tr"].split == "train"
    assert rec_by_id["rec_1_va"].split == "train"
    assert rec_by_id["rec_1_va"].original_split == "val"
    assert rec_by_id["rec_1_va"].split_reassignment_reason is not None

    # Cluster 2: both must be train
    assert rec_by_id["rec_2_tr"].split == "train"
    assert rec_by_id["rec_2_te"].split == "train"
    assert rec_by_id["rec_2_te"].original_split == "test"
    assert rec_by_id["rec_2_te"].split_reassignment_reason is not None

    # Cluster 3: both must be val (VAL > TEST)
    assert rec_by_id["rec_3_va"].split == "val"
    assert rec_by_id["rec_3_te"].split == "val"
    assert rec_by_id["rec_3_te"].original_split == "test"
    assert rec_by_id["rec_3_te"].split_reassignment_reason is not None

    # Clean test singleton stays in test
    assert rec_by_id["rec_clean_test"].split == "test"

    # Verify conflict statistics
    assert stats.affected_clusters_count == 3
    assert stats.reassigned_count == 3
    assert stats.train_val_conflicts == 1
    assert stats.train_test_conflicts == 1
    assert stats.val_test_conflicts == 1
    assert len(stats.reassignments) == 3


def test_splitter_unsupported_strategy():
    """Verify passing an unsupported strategy raises ValueError."""
    import pytest

    records = [
        CanonicalNewsRecord(
            record_id="r1", dataset_name="liar", original_id="1", text="Sample", original_label="true"
        )
    ]
    with pytest.raises(ValueError, match="Unsupported splitting strategy"):
        DatasetSplitter.split_dataset(records, strategy="invalid_strategy")
