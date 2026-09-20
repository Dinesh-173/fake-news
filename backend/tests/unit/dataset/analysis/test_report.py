"""Unit tests for the QualityReportGenerator module."""

from backend.app.dataset.analysis.profiler import DatasetProfiler
from backend.app.dataset.analysis.report import QualityReportGenerator
from backend.app.dataset.schemas import CanonicalNewsRecord


def _make_dummy_records() -> list[CanonicalNewsRecord]:
    return [
        CanonicalNewsRecord(
            record_id=f"rec_{i}",
            dataset_name="dummy_bench",
            original_id=str(i),
            text=f"A test statement {i} containing factual propositions for analysis.",
            original_label="true" if i % 2 == 0 else "false",
            normalized_label="LIKELY_TRUE" if i % 2 == 0 else "LIKELY_FALSE",
            author="Speaker Name",
            metadata={"party_affiliation": "democrat" if i % 2 == 0 else "republican"},
        )
        for i in range(1, 11)
    ]


def test_quality_report_generator(tmp_path):
    records = _make_dummy_records()
    profiler = DatasetProfiler(random_seed=42, git_commit="test_commit")
    profile = profiler.profile_dataset(
        dataset_name="test_report_bench",
        records=records,
        dataset_version="v1.0.0",
    )

    md = QualityReportGenerator.generate_markdown(profile)
    assert "# TruthLens: Dataset Quality Analysis & Statistical Profile" in md
    assert "## 1. Dataset Identity & Provenance" in md
    assert "## 2. Label Distribution & Normalization Coverage" in md
    assert "## 3. Text Statistics & Linguistic Quality Audit" in md
    assert "## 4. Source & Publisher Distribution" in md
    assert "## 5. Temporal Coverage & Longitudinal Stability" in md
    assert "## 6. Metadata Completeness & Attribute Quality" in md
    assert "## 7. Modality Availability" in md
    assert "## 8. Duplicate & Near-Duplicate Analysis" in md
    assert "## 9. Cross-Dataset & Cross-Split Overlap" in md
    assert "## 10. Inferential Statistical Hypothesis Testing" in md
    assert "## 11. Descriptive Research Implications" in md
    assert "## 12. Research Readiness Assessment" in md

    # Verify save
    report_file = tmp_path / "dataset_quality_report.md"
    saved_path = QualityReportGenerator.save_report(profile, report_file)
    assert saved_path.exists()
    assert saved_path.read_text(encoding="utf-8") == md
