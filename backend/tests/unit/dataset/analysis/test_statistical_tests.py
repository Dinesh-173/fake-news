"""Unit tests for the StatisticalTester module."""

from backend.app.dataset.analysis.statistical_tests import StatisticalTester
from backend.app.dataset.schemas import CanonicalNewsRecord


def _make_stat_record(idx: int, label: str, party: str, text: str) -> CanonicalNewsRecord:
    return CanonicalNewsRecord(
        record_id=f"rec_{idx}",
        dataset_name="test_bench",
        original_id=str(idx),
        text=text,
        original_label="label",
        normalized_label=label,
        metadata={"party_affiliation": party},
    )


def test_chi2_party_vs_veracity_sparse():
    # Fewer than 20 records -> returns safely formatted report without crashing
    records = [
        _make_stat_record(1, "LIKELY_TRUE", "democrat", "Statement A"),
        _make_stat_record(2, "LIKELY_FALSE", "republican", "Statement B"),
    ]
    report = StatisticalTester.test_party_vs_veracity(records)
    assert report.sample_size == 2
    assert not report.statistically_significant
    assert "insufficient" in report.limitations[0].lower()


def test_chi2_party_vs_veracity_contingency():
    records = []
    idx = 1
    # Create 30 records: Democrats mostly true, Republicans mostly false
    for _ in range(15):
        records.append(_make_stat_record(idx, "LIKELY_TRUE", "democrat", "Factually accurate point."))
        idx += 1
    for _ in range(5):
        records.append(_make_stat_record(idx, "LIKELY_FALSE", "democrat", "Inaccurate point."))
        idx += 1
    for _ in range(5):
        records.append(_make_stat_record(idx, "LIKELY_TRUE", "republican", "Factually accurate point."))
        idx += 1
    for _ in range(15):
        records.append(_make_stat_record(idx, "LIKELY_FALSE", "republican", "Inaccurate point."))
        idx += 1

    report = StatisticalTester.test_party_vs_veracity(records)
    assert report.sample_size == 40
    assert report.test_statistic_value > 0.0
    assert 0.0 <= report.effect_size_value <= 1.0
    assert report.p_value < 0.05  # Strongly biased distribution should reject H0
    assert report.statistically_significant


def test_mann_whitney_u_text_length():
    records = []
    # Group 1: short statements (len ~ 20)
    for i in range(20):
        records.append(_make_stat_record(i, "LIKELY_TRUE", "none", "Short text sample."))
    # Group 2: long statements (len ~ 100)
    for i in range(20, 40):
        records.append(_make_stat_record(i, "LIKELY_FALSE", "none", "A substantially longer statement designed to test the Mann-Whitney rank comparison algorithm."))

    report = StatisticalTester.test_text_length_by_veracity(
        records, group1_label="LIKELY_TRUE", group2_label="LIKELY_FALSE"
    )
    assert report.sample_size == 40
    assert report.statistically_significant
    assert report.p_value < 1e-4
    assert abs(report.effect_size_value) > 0.5


def test_label_uniformity_goodness_of_fit():
    # Exactly equal counts across 3 classes: 10 each
    records = []
    for i in range(10):
        records.append(_make_stat_record(i, "LIKELY_TRUE", "none", "Text"))
    for i in range(10, 20):
        records.append(_make_stat_record(i, "LIKELY_FALSE", "none", "Text"))
    for i in range(20, 30):
        records.append(_make_stat_record(i, "MISLEADING_OR_PARTIALLY_SUPPORTED", "none", "Text"))

    report = StatisticalTester.test_label_uniformity(records)
    assert report.sample_size == 30
    assert report.degrees_of_freedom == 2
    assert report.test_statistic_value == 0.0  # Perfect uniformity
    assert report.p_value == 1.0
    assert not report.statistically_significant
