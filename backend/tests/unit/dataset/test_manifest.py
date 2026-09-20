"""Unit tests for dataset manifest and report generation."""

import json
from pathlib import Path

from backend.app.dataset.manifests.manifest import DatasetManifestGenerator
from backend.app.dataset.reporting.quality_report import QualityReporter
from backend.app.dataset.schemas import (
    CanonicalNewsRecord,
    DuplicateStatistics,
    SplitStatistics,
    ValidationSummary,
)


def test_manifest_generation_and_serialization(tmp_path: Path):
    """Verify manifest creation, statistical calculations, and JSON export."""
    records = [
        CanonicalNewsRecord(
            record_id="r1",
            dataset_name="liar",
            original_id="1",
            text="Statement 1",
            original_label="true",
            normalized_label="LIKELY_TRUE",
            has_text=True,
            has_image=False,
            split="train",
        ),
        CanonicalNewsRecord(
            record_id="r2",
            dataset_name="liar",
            original_id="2",
            text="Statement 2",
            original_label="false",
            normalized_label="LIKELY_FALSE",
            has_text=True,
            has_image=True,
            split="test",
        ),
    ]

    val_summary = ValidationSummary(
        total_records=2, valid_records=2, warning_records=0, invalid_records=0
    )
    dup_stats = DuplicateStatistics(
        total_records=2,
        unique_content_hashes=2,
        exact_duplicates_count=0,
        exact_duplicate_groups=0,
        near_duplicate_groups=0,
    )
    split_stats = SplitStatistics(
        train_count=1,
        val_count=0,
        test_count=1,
        train_distribution={"LIKELY_TRUE": 1},
        test_distribution={"LIKELY_FALSE": 1},
    )

    manifest = DatasetManifestGenerator.generate_manifest(
        dataset_name="liar",
        dataset_version="v1.0.0",
        source_url_or_origin="Test Source",
        license_or_terms="MIT",
        records=records,
        validation_summary=val_summary,
        duplicate_stats=dup_stats,
        split_stats=split_stats,
    )

    manifest_path = tmp_path / "manifest.json"
    DatasetManifestGenerator.save_manifest(manifest, manifest_path)

    assert manifest_path.exists()
    with open(manifest_path, encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["dataset_name"] == "liar"
    assert loaded["record_counts"]["total"] == 2
    assert loaded["modality_statistics"]["image_available"] == 1

    # Test Markdown report generation
    report_content = QualityReporter.generate_markdown_report(manifest)
    assert "# TruthLens Dataset Quality Report: LIAR" in report_content
    assert "LIKELY_TRUE" in report_content
    assert "LIKELY_FALSE" in report_content
