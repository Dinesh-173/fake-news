"""Master dataset pipeline orchestrating validation, cleaning, deduplication, and export."""

import json
from pathlib import Path

from backend.app.core.logging import logger
from backend.app.dataset.cleaning.cleaner import DataCleaner
from backend.app.dataset.deduplication.deduplicator import Deduplicator
from backend.app.dataset.loaders.base import BaseDatasetLoader
from backend.app.dataset.manifests.manifest import DatasetManifestGenerator
from backend.app.dataset.normalization.labels import LabelNormalizer
from backend.app.dataset.normalization.sources import SourceNormalizer
from backend.app.dataset.reporting.quality_report import QualityReporter
from backend.app.dataset.schemas import CanonicalNewsRecord, DatasetManifest
from backend.app.dataset.splitting.splitter import DatasetSplitter
from backend.app.dataset.validation.validator import DataValidator


class DatasetPipeline:
    """Orchestrates the end-to-end dataset transformation, audit, and export lifecycle."""

    def __init__(
        self,
        loader: BaseDatasetLoader,
        label_mappings_dir: Path | None = None,
        random_seed: int = 42,
    ):
        self.loader = loader
        self.label_normalizer = LabelNormalizer(mappings_dir=label_mappings_dir)
        self.random_seed = random_seed

    def run_pipeline(
        self,
        source_path: Path,
        output_dir: Path,
        dataset_version: str = "v1.0.0",
        source_url_or_origin: str = "Academic Repository",
        license_or_terms: str = "Academic Research Only",
        split_strategy: str = "stratified",
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        near_duplicate_threshold: float = 0.85,
    ) -> tuple[list[CanonicalNewsRecord], DatasetManifest, Path]:
        """
        Executes full pipeline:
        Raw -> Validation -> Cleaning -> Deduplication -> Normalization -> Splitting -> Export -> Manifest -> Quality Report
        """
        logger.info(
            "Starting dataset pipeline for '%s' (%s)...", self.loader.dataset_name, dataset_version
        )

        # 1. Ingestion
        raw_records = self.loader.load_records(source_path)
        logger.info("Step 1/7 Ingestion complete: %d records loaded.", len(raw_records))

        # 2. Validation
        validated_records, val_summary = DataValidator.validate_dataset(raw_records)
        logger.info(
            "Step 2/7 Validation complete: %d valid, %d warning, %d invalid.",
            val_summary.valid_records,
            val_summary.warning_records,
            val_summary.invalid_records,
        )

        # 3. Cleaning
        cleaned_records = [DataCleaner.clean_record(r) for r in validated_records]
        logger.info("Step 3/7 Cleaning complete.")

        # 4. Deduplication & Cluster Identification
        deduped_records, dup_stats = Deduplicator.process_duplicates(
            cleaned_records,
            near_duplicate_threshold=near_duplicate_threshold,
        )
        logger.info(
            "Step 4/7 Deduplication complete: %d exact dups (%d groups), %d near dup groups.",
            dup_stats.exact_duplicates_count,
            dup_stats.exact_duplicate_groups,
            dup_stats.near_duplicate_groups,
        )

        # 5. Label & Source Normalization
        normalized_records: list[CanonicalNewsRecord] = []
        for r in deduped_records:
            r_norm = self.label_normalizer.normalize_record(r)
            domain = SourceNormalizer.extract_domain(r_norm.source_url or r_norm.source_name_raw)
            norm_source = SourceNormalizer.normalize_source_name(r_norm.source_name_raw)
            normalized_records.append(
                r_norm.model_copy(
                    update={
                        "source_domain": domain,
                        "source_name_normalized": norm_source or r_norm.source_name_normalized,
                    }
                )
            )
        logger.info("Step 5/7 Label & source normalization complete.")

        # 6. Leakage-Free Splitting
        final_records, split_stats = DatasetSplitter.split_dataset(
            normalized_records,
            strategy=split_strategy,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=self.random_seed,
        )
        logger.info(
            "Step 6/7 Splitting complete: Train=%d, Val=%d, Test=%d.",
            split_stats.train_count,
            split_stats.val_count,
            split_stats.test_count,
        )

        # 7. Export & Artifact Emission
        output_dir.mkdir(parents=True, exist_ok=True)
        all_jsonl_path = output_dir / "all_records.jsonl"
        train_jsonl_path = output_dir / "train.jsonl"
        val_jsonl_path = output_dir / "val.jsonl"
        test_jsonl_path = output_dir / "test.jsonl"

        with (
            open(all_jsonl_path, "w", encoding="utf-8") as f_all,
            open(train_jsonl_path, "w", encoding="utf-8") as f_tr,
            open(val_jsonl_path, "w", encoding="utf-8") as f_va,
            open(test_jsonl_path, "w", encoding="utf-8") as f_te,
        ):
            for r in final_records:
                line = json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n"
                f_all.write(line)
                if r.split == "train":
                    f_tr.write(line)
                elif r.split == "val":
                    f_va.write(line)
                elif r.split == "test":
                    f_te.write(line)

        artifact_files = {
            "source_raw": source_path,
            "all_records": all_jsonl_path,
            "train_split": train_jsonl_path,
            "val_split": val_jsonl_path,
            "test_split": test_jsonl_path,
        }
        if source_path.is_dir():
            for f in sorted(source_path.iterdir()):
                if f.is_file():
                    artifact_files[f"source_raw_{f.name}"] = f

        # 8. Manifest & Quality Report
        manifest = DatasetManifestGenerator.generate_manifest(
            dataset_name=self.loader.dataset_name,
            dataset_version=dataset_version,
            source_url_or_origin=source_url_or_origin,
            license_or_terms=license_or_terms,
            records=final_records,
            validation_summary=val_summary,
            duplicate_stats=dup_stats,
            split_stats=split_stats,
            random_seed=self.random_seed,
            artifact_files=artifact_files,
        )

        manifest_path = output_dir / "manifest.json"
        DatasetManifestGenerator.save_manifest(manifest, manifest_path)

        report_path = output_dir / "quality_report.md"
        QualityReporter.save_report(manifest, report_path)

        logger.info("Step 7/7 Pipeline finished successfully. Artifacts saved in %s", output_dir)
        return final_records, manifest, report_path
