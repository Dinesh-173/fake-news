"""Integration test executing the full dataset pipeline end-to-end on synthetic fixtures."""

import json
from pathlib import Path

from backend.app.dataset.loaders.liar import LiarDatasetLoader
from backend.app.dataset.pipeline import DatasetPipeline


def test_full_dataset_pipeline_integration(tmp_path: Path):
    """
    Executes full pipeline:
    Raw TSV -> Load -> Validate -> Clean -> Deduplicate -> Normalize -> Split -> Export -> Manifest -> Report
    """
    raw_fixture_path = tmp_path / "raw_sample_liar.tsv"
    fixture_rows = [
        "2001.json\ttrue\tNASA launched the exploration rover to Mars successfully.\tspace\tNASA Official\tspokesperson\tDC\tnone\t0\t0\t0\t1\t0\tofficial release",
        "2002.json\tfalse\tThe moon is artificially hollow and contains alien machinery.\tconspiracy\tAnonymous\tnone\tnone\tnone\t0\t5\t0\t0\t2\tblog post",
        "2003.json\thalf-true\tGross domestic product increased by 3 percent this calendar year.\teconomy\tEconomist\tanalyst\tNY\tnone\t1\t1\t4\t2\t0\tfinancial interview",
        "2004.json\tmostly-true\tVaccination rates rose sharply across all major metropolitan regions.\thealth\tHealth Director\tdoctor\tGA\tnone\t0\t0\t1\t4\t0\tpress briefing",
        "2005.json\tpants-fire\tSunlight has been banned by the federal government under new law.\tlaw\tSatirical Site\twriter\tCA\tnone\t0\t0\t0\t0\t10\tparody forum",
        "2006.json\tbarely-true\tNew solar subsidies will completely eliminate energy bills for all.\tenergy\tPromoter\tsales\tAZ\tnone\t3\t2\t1\t0\t0\tad video",
        "2007.json\tfalse\tThe moon is artificially hollow and contains alien machinery.\tconspiracy\tAnonymous 2\tnone\tnone\tnone\t0\t5\t0\t0\t2\twire copy",  # Exact duplicate of 2002
        "2008.json\ttrue\tNASA launched the exploration rover to Mars successfully today.\tspace\tNASA Media\tspokesperson\tDC\tnone\t0\t0\t0\t1\t0\tnews brief",  # Near duplicate of 2001
        "2009.json\ttrue\tUnemployment dropped to historic lows according to recent statistics.\teconomy\tLabor Sec\tofficial\tDC\tnone\t0\t0\t1\t3\t0\tspeech",
        "2010.json\tfalse\tA single pill cures all human illnesses within ten minutes.\thealth\tSnake Oil\tpromoter\tNV\tnone\t0\t8\t0\t0\t3\tsocial ad",
    ]
    raw_fixture_path.write_text("\n".join(fixture_rows) + "\n", encoding="utf-8")

    output_dir = tmp_path / "processed_liar"
    pipeline = DatasetPipeline(loader=LiarDatasetLoader(), random_seed=42)

    records, manifest, report_path = pipeline.run_pipeline(
        source_path=raw_fixture_path,
        output_dir=output_dir,
        dataset_version="v1.0-test",
        source_url_or_origin="TruthLens Synthetic Test Fixture",
        license_or_terms="Test Use Only",
        split_strategy="stratified",
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    # 1. Verify record count
    assert len(records) == 10

    # 2. Verify validation and cleaning
    assert all(r.validation_status in ("VALID", "WARNING") for r in records)
    assert all(r.has_text for r in records)

    # 3. Verify deduplication
    assert manifest.duplicate_statistics.exact_duplicates_count == 1
    assert manifest.duplicate_statistics.exact_duplicate_groups == 1

    dup_record = next(r for r in records if r.original_id == "2007.json")
    orig_record = next(r for r in records if r.original_id == "2002.json")
    assert dup_record.is_duplicate is True
    assert dup_record.duplicate_group_id is not None
    assert dup_record.duplicate_group_id == orig_record.duplicate_group_id

    # 4. Critical Leakage Check: Duplicate records MUST be in the same split
    assert dup_record.split == orig_record.split, (
        f"Leakage violation! Primary in {orig_record.split}, duplicate in {dup_record.split}"
    )

    # 5. Verify Label Normalization
    assert orig_record.normalized_label == "LIKELY_FALSE"
    true_rec = next(r for r in records if r.original_id == "2001.json")
    assert true_rec.normalized_label == "LIKELY_TRUE"

    # 6. Verify File Exports
    assert (output_dir / "all_records.jsonl").exists()
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "val.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "quality_report.md").exists()

    # 7. Check manifest JSON contents
    with open(output_dir / "manifest.json", encoding="utf-8") as f:
        m_data = json.load(f)
    assert m_data["dataset_name"] == "liar"
    assert m_data["record_counts"]["total"] == 10
    assert "source_raw" in m_data["file_hashes"]
    assert "all_records" in m_data["file_hashes"]

    # 8. Check Quality Report
    report_text = report_path.read_text(encoding="utf-8")
    assert "# TruthLens Dataset Quality Report: LIAR" in report_text
    assert "Unique Content Hashes (SHA-256)" in report_text
