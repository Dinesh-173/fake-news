"""Unit tests for dataset canonical schema and statistical models."""

from backend.app.dataset.schemas import (
    CanonicalNewsRecord,
    ModalityStatistics,
    ValidationSummary,
)


def test_canonical_record_creation():
    """Verify default values and field assignment in CanonicalNewsRecord."""
    rec = CanonicalNewsRecord(
        record_id="rec_001",
        dataset_name="liar",
        original_id="101.json",
        text="Sample news statement asserting a verifiable fact.",
        original_label="false",
    )
    assert rec.record_id == "rec_001"
    assert rec.dataset_name == "liar"
    assert rec.original_label == "false"
    assert rec.normalized_label is None
    assert rec.validation_status == "VALID"
    assert rec.is_duplicate is False
    assert rec.has_text is True
    assert rec.has_image is False


def test_modality_statistics():
    """Verify modality statistics structure."""
    ms = ModalityStatistics(
        total=100,
        text_available=100,
        image_available=40,
        text_and_image_available=40,
        metadata_available=85,
        timestamp_available=90,
        source_available=95,
    )
    assert ms.total == 100
    assert ms.text_available == 100
    assert ms.image_available == 40


def test_validation_summary():
    """Verify validation summary counters."""
    vs = ValidationSummary(
        total_records=50,
        valid_records=45,
        warning_records=3,
        invalid_records=2,
        warning_reasons={"extremely_short_text": 3},
        invalid_reasons={"empty_or_whitespace_text": 2},
    )
    assert vs.total_records == 50
    assert vs.warning_reasons["extremely_short_text"] == 3
