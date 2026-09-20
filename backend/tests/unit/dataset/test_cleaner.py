"""Unit tests for minimally destructive text and timestamp cleaner."""

from backend.app.dataset.cleaning.cleaner import DataCleaner
from backend.app.dataset.schemas import CanonicalNewsRecord


def test_clean_text_unicode_and_whitespace():
    """Verify Unicode normalization and whitespace collapsing while preserving punctuation."""
    raw = "Breaking  News:\tScientists   discover   new\n\n\n\nsubstance!  "
    cleaned = DataCleaner.clean_text(raw)
    assert cleaned == "Breaking News: Scientists discover new\n\nsubstance!"
    # Ensure punctuation, casing, exclamation mark are preserved
    assert "Breaking News:" in cleaned
    assert "substance!" in cleaned


def test_clean_text_strips_control_characters():
    """Verify unprintable control characters are stripped."""
    raw = "Clean\x00 text\x07 with\x1b control\x08 characters"
    cleaned = DataCleaner.clean_text(raw)
    assert cleaned == "Clean text with control characters"


def test_normalize_timestamp():
    """Verify standard date string parsing to UTC."""
    raw_ts = "2024-05-12 14:30:00-05:00"
    iso_str, dt = DataCleaner.normalize_timestamp(raw_ts)
    assert dt is not None
    assert dt.year == 2024
    assert dt.month == 5
    assert dt.day == 12

    # Empty timestamp returns None
    assert DataCleaner.normalize_timestamp("") == (None, None)
    assert DataCleaner.normalize_timestamp(None) == (None, None)


def test_clean_record_updates_modality_flags():
    """Verify record cleaning updates text and modality flags."""
    record = CanonicalNewsRecord(
        record_id="r1",
        dataset_name="fakenewsnet",
        original_id="fn1",
        text="   Valid  article text for cleaning.   ",
        image_url="https://example.com/photo.jpg",
        published_at_raw="2023-01-01T00:00:00Z",
        original_label="real",
    )
    cleaned = DataCleaner.clean_record(record)
    assert cleaned.text == "Valid article text for cleaning."
    assert cleaned.has_text is True
    assert cleaned.has_image is True
    assert cleaned.has_timestamp is True
    assert cleaned.published_at_normalized is not None
