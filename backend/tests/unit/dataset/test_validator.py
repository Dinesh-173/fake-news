"""Unit tests for dataset record validation rules."""

from backend.app.dataset.schemas import CanonicalNewsRecord
from backend.app.dataset.validation.validator import DataValidator


def test_validate_valid_record():
    """Verify that a compliant record receives VALID status."""
    record = CanonicalNewsRecord(
        record_id="r1",
        dataset_name="liar",
        original_id="101.json",
        text="A sufficiently detailed statement for verification.",
        original_label="true",
    )
    status, notes = DataValidator.validate_record(record)
    assert status == "VALID"
    assert len(notes) == 0


def test_validate_empty_text_is_invalid():
    """Verify empty or whitespace-only text is classified as INVALID."""
    record = CanonicalNewsRecord(
        record_id="r2",
        dataset_name="liar",
        original_id="102.json",
        text="     ",
        original_label="false",
    )
    status, notes = DataValidator.validate_record(record)
    assert status == "INVALID"
    assert "empty_or_whitespace_text" in notes


def test_validate_extremely_short_text_is_warning():
    """Verify short text triggers WARNING rather than silent dropping."""
    record = CanonicalNewsRecord(
        record_id="r3",
        dataset_name="liar",
        original_id="103.json",
        text="Tiny",
        original_label="half-true",
    )
    status, notes = DataValidator.validate_record(record)
    assert status == "WARNING"
    assert any("short_text" in n for n in notes)


def test_validate_dataset_batch():
    """Verify batch validation returns updated records and summary."""
    records = [
        CanonicalNewsRecord(
            record_id="r1",
            dataset_name="liar",
            original_id="1",
            text="Valid length statement here.",
            original_label="true",
        ),
        CanonicalNewsRecord(
            record_id="r2",
            dataset_name="liar",
            original_id="2",
            text="",
            original_label="false",
        ),
    ]
    validated, summary = DataValidator.validate_dataset(records)
    assert summary.total_records == 2
    assert summary.valid_records == 1
    assert summary.invalid_records == 1
    assert "empty_or_whitespace_text" in summary.invalid_reasons
