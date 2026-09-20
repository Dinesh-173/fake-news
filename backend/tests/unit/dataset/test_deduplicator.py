"""Unit tests for exact and near-duplicate detection."""

from backend.app.dataset.deduplication.deduplicator import Deduplicator
from backend.app.dataset.schemas import CanonicalNewsRecord


def test_content_hash_deterministic():
    """Verify content hash is invariant to extra whitespace and casing."""
    t1 = "Breaking News: Economy grew by 3 percent."
    t2 = "  breaking   news: economy grew by 3 percent.  "
    h1 = Deduplicator.compute_content_hash(t1)
    h2 = Deduplicator.compute_content_hash(t2)
    assert h1 == h2
    assert len(h1) == 64


def test_exact_duplicate_detection_and_grouping():
    """Verify exact duplicates are identified, grouped, and primary record retained."""
    records = [
        CanonicalNewsRecord(
            record_id="r1",
            dataset_name="liar",
            original_id="1",
            text="Federal deficit reached a new high this quarter.",
            original_label="false",
            title="Article 1",
        ),
        CanonicalNewsRecord(
            record_id="r2",
            dataset_name="liar",
            original_id="2",
            text="FEDERAL DEFICIT REACHED A NEW HIGH THIS QUARTER.  ",
            original_label="pants-fire",
        ),
        CanonicalNewsRecord(
            record_id="r3",
            dataset_name="liar",
            original_id="3",
            text="Unrelated unique statement about space exploration.",
            original_label="true",
        ),
    ]

    deduped, stats = Deduplicator.process_duplicates(records)
    assert stats.total_records == 3
    assert stats.unique_content_hashes == 2
    assert stats.exact_duplicates_count == 1
    assert stats.exact_duplicate_groups == 1

    # Find r1 and r2 in results
    r1_res = next(r for r in deduped if r.record_id == "r1")
    r2_res = next(r for r in deduped if r.record_id == "r2")
    r3_res = next(r for r in deduped if r.record_id == "r3")

    assert r1_res.duplicate_group_id is not None
    assert r1_res.duplicate_group_id == r2_res.duplicate_group_id
    assert r3_res.duplicate_group_id is None

    # One is retained primary, one is duplicate
    assert (r1_res.is_duplicate and not r2_res.is_duplicate) or (
        r2_res.is_duplicate and not r1_res.is_duplicate
    )


def test_near_duplicate_detection():
    """Verify near-identical texts are tagged with near_duplicate_group_id."""
    records = [
        CanonicalNewsRecord(
            record_id="n1",
            dataset_name="liar",
            original_id="n1",
            text="President announced new infrastructure funding in Ohio speech on Monday.",
            original_label="true",
        ),
        CanonicalNewsRecord(
            record_id="n2",
            dataset_name="liar",
            original_id="n2",
            text="President announced new infrastructure funding in Ohio speech on Tuesday.",
            original_label="mostly-true",
        ),
    ]
    deduped, stats = Deduplicator.process_duplicates(records, near_duplicate_threshold=0.80)
    assert stats.near_duplicate_groups == 1
    assert deduped[0].near_duplicate_group_id is not None
    assert deduped[0].near_duplicate_group_id == deduped[1].near_duplicate_group_id
