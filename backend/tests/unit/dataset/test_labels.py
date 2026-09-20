"""Unit tests for dataset label normalization and provenance preservation."""

from backend.app.dataset.normalization.labels import LabelNormalizer
from backend.app.dataset.schemas import CanonicalNewsRecord


def test_liar_label_mappings():
    """Verify all 6 LIAR labels map correctly to TruthLens states."""
    normalizer = LabelNormalizer()

    mappings = {
        "pants-fire": "LIKELY_FALSE",
        "false": "LIKELY_FALSE",
        "barely-true": "MISLEADING_OR_PARTIALLY_SUPPORTED",
        "half-true": "MISLEADING_OR_PARTIALLY_SUPPORTED",
        "mostly-true": "LIKELY_TRUE",
        "true": "LIKELY_TRUE",
    }

    for raw, expected in mappings.items():
        target, status = normalizer.normalize_label("liar", raw)
        assert target == expected
        assert status == "mapped"


def test_unresolved_labels_handled_safely():
    """Verify unmapped labels receive unresolved status without crashing or guessing."""
    normalizer = LabelNormalizer()
    target, status = normalizer.normalize_label("liar", "completely_unknown_label")
    assert target is None
    assert status == "unresolved"


def test_normalize_record_preserves_original():
    """Verify that record normalization preserves the original label verbatim."""
    normalizer = LabelNormalizer()
    record = CanonicalNewsRecord(
        record_id="r1",
        dataset_name="fakenewsnet",
        original_id="fn1",
        text="A real news story headline.",
        original_label="real",
    )
    normalized = normalizer.normalize_record(record)
    assert normalized.original_label == "real"
    assert normalized.normalized_label == "LIKELY_TRUE"
    assert normalized.label_mapping_status == "mapped"
