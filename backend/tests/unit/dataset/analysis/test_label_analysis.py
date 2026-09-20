"""Unit tests for the LabelAnalyzer module."""

from backend.app.dataset.analysis.label_analysis import LabelAnalyzer
from backend.app.dataset.schemas import CanonicalNewsRecord


def _make_record(idx: int, orig_label: str, norm_label: str | None, status: str = "mapped") -> CanonicalNewsRecord:
    return CanonicalNewsRecord(
        record_id=f"rec_{idx}",
        dataset_name="test_bench",
        original_id=str(idx),
        text="Sample claim statement text for verification.",
        original_label=orig_label,
        normalized_label=norm_label,
        label_mapping_status=status,
    )


def test_label_analyzer_empty():
    res = LabelAnalyzer.analyze([])
    assert res.total_records == 0
    assert res.unique_original_labels == 0
    assert res.mapping_coverage == 0.0
    assert res.majority_class == "NONE"
    assert res.normalized_entropy == 0.0


def test_label_analyzer_balanced():
    records = [
        _make_record(1, "true", "LIKELY_TRUE"),
        _make_record(2, "true", "LIKELY_TRUE"),
        _make_record(3, "false", "LIKELY_FALSE"),
        _make_record(4, "false", "LIKELY_FALSE"),
    ]
    res = LabelAnalyzer.analyze(records)
    assert res.total_records == 4
    assert res.unique_original_labels == 2
    assert res.unique_normalized_labels == 2
    assert res.original_distribution["true"] == 2
    assert res.original_distribution["false"] == 2
    assert res.mapping_coverage == 1.0
    assert res.imbalance_ratio == 1.0
    # Two equal classes -> normalized entropy should be 1.0
    assert abs(res.normalized_entropy - 1.0) < 1e-3


def test_label_analyzer_unmapped_and_imbalance():
    records = [
        _make_record(1, "pants-fire", "LIKELY_FALSE"),
        _make_record(2, "pants-fire", "LIKELY_FALSE"),
        _make_record(3, "pants-fire", "LIKELY_FALSE"),
        _make_record(4, "true", "LIKELY_TRUE"),
        _make_record(5, "unknown_verdict", None, status="unresolved"),
    ]
    res = LabelAnalyzer.analyze(records)
    assert res.total_records == 5
    assert res.unresolved_count == 1
    assert "unknown_verdict" in res.unresolved_labels
    assert res.mapping_coverage == 0.8
    assert res.majority_class == "LIKELY_FALSE"
    assert res.majority_class_percentage == 60.0
    assert res.minority_class == "LIKELY_TRUE"
    assert res.imbalance_ratio == 3.0  # 3 LIKELY_FALSE vs 1 LIKELY_TRUE
    assert len(res.mapping_breakdown) == 3
