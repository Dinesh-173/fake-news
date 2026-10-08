"""Dataset manifest generator recording provenance, file hashes, and distributions."""

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from backend.app.core.logging import logger
from backend.app.dataset.schemas import (
    CanonicalNewsRecord,
    DatasetManifest,
    DuplicateStatistics,
    ModalityStatistics,
    SplitStatistics,
    ValidationSummary,
)


class DatasetManifestGenerator:
    """Produces cryptographic and statistical manifest for a processed dataset artifact."""

    @classmethod
    def compute_file_sha256(cls, file_path: Path) -> str:
        """Computes SHA-256 hash of a file on disk."""
        if not file_path.exists():
            return "FILE_NOT_FOUND"
        if file_path.is_dir():
            return "DIRECTORY"
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @classmethod
    def get_git_commit(cls) -> str:
        """Retrieves the current git commit hash."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return res.stdout.strip()
        except Exception:
            return "UNKNOWN_COMMIT"

    @classmethod
    def generate_manifest(
        cls,
        dataset_name: str,
        dataset_version: str,
        source_url_or_origin: str,
        license_or_terms: str,
        records: list[CanonicalNewsRecord],
        validation_summary: ValidationSummary,
        duplicate_stats: DuplicateStatistics,
        split_stats: SplitStatistics,
        random_seed: int = 42,
        artifact_files: dict[str, Path] | None = None,
    ) -> DatasetManifest:
        """Computes all summary statistics and constructs the manifest."""
        # 1. Modality statistics
        total = len(records)
        text_avail = sum(1 for r in records if r.has_text)
        img_avail = sum(1 for r in records if r.has_image)
        both_avail = sum(1 for r in records if r.has_text and r.has_image)
        meta_avail = sum(1 for r in records if r.has_metadata)
        ts_avail = sum(1 for r in records if r.has_timestamp)
        src_avail = sum(1 for r in records if r.has_source)

        mod_stats = ModalityStatistics(
            total=total,
            text_available=text_avail,
            image_available=img_avail,
            text_and_image_available=both_avail,
            metadata_available=meta_avail,
            timestamp_available=ts_avail,
            source_available=src_avail,
        )

        # 2. Label distributions
        orig_counter = Counter(r.original_label for r in records)
        norm_counter = Counter(r.normalized_label or "UNRESOLVED" for r in records)

        # 3. File hashes
        file_hashes: dict[str, str] = {}
        if artifact_files:
            for name, path in artifact_files.items():
                file_hashes[name] = cls.compute_file_sha256(path)

        manifest = DatasetManifest(
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            pipeline_version="1.0.0",
            created_at=datetime.now(timezone.utc).isoformat(),
            source_url_or_origin=source_url_or_origin,
            license_or_terms=license_or_terms,
            record_counts={
                "total": total,
                "valid": validation_summary.valid_records,
                "warning": validation_summary.warning_records,
                "invalid": validation_summary.invalid_records,
            },
            label_distribution_original=dict(orig_counter),
            label_distribution_normalized=dict(norm_counter),
            modality_statistics=mod_stats,
            duplicate_statistics=duplicate_stats,
            split_statistics=split_stats,
            random_seed=random_seed,
            git_commit=cls.get_git_commit(),
            file_hashes=file_hashes,
        )

        return manifest

    @classmethod
    def save_manifest(cls, manifest: DatasetManifest, output_path: Path) -> Path:
        """Saves the manifest model as formatted JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(), f, indent=2)
        logger.info("Saved dataset manifest to %s", output_path)
        return output_path
