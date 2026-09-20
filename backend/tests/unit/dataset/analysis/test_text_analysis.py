"""Unit tests for the TextAnalyzer module."""

from backend.app.dataset.analysis.text_analysis import TextAnalyzer, TextQualityThresholds
from backend.app.dataset.schemas import CanonicalNewsRecord


def _make_text_record(idx: int, text: str, norm_label: str = "LIKELY_TRUE") -> CanonicalNewsRecord:
    return CanonicalNewsRecord(
        record_id=f"rec_{idx}",
        dataset_name="test_bench",
        original_id=str(idx),
        text=text,
        original_label="true",
        normalized_label=norm_label,
    )


def test_text_analyzer_tokenizer():
    tokens = TextAnalyzer.tokenize("NASA's Rover landed on Mars in 2021! It's working.")
    assert "NASA's" in tokens
    assert "landed" in tokens
    assert "!" in tokens
    assert "It's" in tokens
    assert "." in tokens


def test_text_analyzer_empty_and_whitespace():
    records = [
        _make_text_record(1, ""),
        _make_text_record(2, "   \n\t  "),
        _make_text_record(3, "A valid statement with sufficient character length."),
    ]
    res = TextAnalyzer.analyze(records)
    assert res.total_records == 3
    assert res.quality_flags.empty_count == 1
    assert res.quality_flags.whitespace_only_count == 1
    assert res.quality_flags.total_flagged_records >= 2


def test_text_analyzer_distributions_and_percentiles():
    # 10 records with predictable lengths: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 characters
    records = [
        _make_text_record(i, "x" * (i * 10))
        for i in range(1, 11)
    ]
    res = TextAnalyzer.analyze(records)
    assert res.char_length_stats.count == 10
    assert res.char_length_stats.min == 10.0
    assert res.char_length_stats.max == 100.0
    assert res.char_length_stats.mean == 55.0
    assert res.char_length_stats.median == 55.0
    assert res.char_length_stats.p25 <= res.char_length_stats.p50 <= res.char_length_stats.p75


def test_text_analyzer_quality_flags():
    thresh = TextQualityThresholds(
        min_char_length=15,
        max_char_length=100,
        max_punct_ratio=0.25,
        max_url_count=1,
        repetition_max_ratio=0.50,
    )
    records = [
        _make_text_record(1, "Too short"),  # < 15 chars
        _make_text_record(2, "Long " * 30),  # > 100 chars
        _make_text_record(3, "Spam!?!?!?!?!? $$$$$$ ****** !!!!!!"),  # High punctuation
        _make_text_record(4, "Check this https://a.com and also http://b.com and www.c.com"),  # Excessive URLs
        _make_text_record(5, "word word word word word word word word word word"),  # Low vocabulary density
        _make_text_record(6, "The governor signed the new infrastructure funding bill into state law yesterday."),  # Normal
    ]
    res = TextAnalyzer.analyze(records, thresholds=thresh)
    qf = res.quality_flags
    assert qf.extremely_short_count == 1
    assert qf.extremely_long_count == 1
    assert qf.excessive_punctuation_count >= 1
    assert qf.excessive_urls_count == 1
    assert qf.repetitive_text_count >= 1
    assert qf.total_flagged_records == 5
