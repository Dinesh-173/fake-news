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
