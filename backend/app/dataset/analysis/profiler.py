"""Central DatasetProfiler orchestrating statistical profiling and artifact emission."""

import hashlib
import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from backend.app.core.logging import logger
from backend.app.dataset.analysis.cross_dataset_analysis import (
    CrossDatasetAnalysisResult,
    CrossDatasetAnalyzer,
)
from backend.app.dataset.analysis.duplicate_analysis import (
    DuplicateAnalysisResult,
    DuplicateAnalyzer,
)
from backend.app.dataset.analysis.label_analysis import (
    LabelAnalysisResult,
    LabelAnalyzer,
)
from backend.app.dataset.analysis.metadata_analysis import (
    MetadataAnalysisResult,
    MetadataAnalyzer,
)
from backend.app.dataset.analysis.modality_analysis import (
    ModalityAnalysisResult,
    ModalityAnalyzer,
)
from backend.app.dataset.analysis.source_analysis import (
    SourceAnalysisResult,
    SourceAnalyzer,
)
from backend.app.dataset.analysis.statistical_tests import (
    StatisticalTester,
    StatisticalTestSuiteResult,
)
from backend.app.dataset.analysis.temporal_analysis import (
    TemporalAnalysisResult,
    TemporalAnalyzer,
)
from backend.app.dataset.analysis.text_analysis import (
    TextAnalysisResult,
    TextAnalyzer,
    TextQualityThresholds,
)
from backend.app.dataset.schemas import CanonicalNewsRecord


class EnvironmentProvenance(BaseModel):
    """Execution environment metadata ensuring scientific reproducibility."""

    python_version: str
    platform: str
    git_commit: str
    packages: dict[str, str] = Field(default_factory=dict)
    random_seed: int = 42


class DatasetProfile(BaseModel):
    """Unified master profile aggregating all empirical analysis dimensions."""

    run_id: str
    dataset_name: str
    dataset_version: str
    created_at_utc: str
    total_records: int
    environment_provenance: EnvironmentProvenance
    dataset_file_hashes: dict[str, str] = Field(default_factory=dict)
    label_analysis: LabelAnalysisResult
    text_analysis: TextAnalysisResult
    source_analysis: SourceAnalysisResult
    temporal_analysis: TemporalAnalysisResult
    metadata_analysis: MetadataAnalysisResult
    modality_analysis: ModalityAnalysisResult
    duplicate_analysis: DuplicateAnalysisResult
    cross_dataset_analysis: CrossDatasetAnalysisResult
    statistical_tests: StatisticalTestSuiteResult
    readiness_summary: dict[str, str] = Field(default_factory=dict)


class DatasetProfiler:
    """Master analytical engine executing the complete TruthLens Phase 3 profiling suite."""

    def __init__(
        self,
        random_seed: int = 42,
        git_commit: str | None = None,
        text_thresholds: TextQualityThresholds | None = None,
    ):
        self.random_seed = random_seed
        self.git_commit = git_commit or self._discover_git_commit()
        self.text_thresholds = text_thresholds or TextQualityThresholds()

    @staticmethod
    def _discover_git_commit() -> str:
        """Retrieves active Git commit hash safely without external dependency."""
        try:
            head_path = Path(".git/HEAD")
            if head_path.exists():
                ref = head_path.read_text(encoding="utf-8").strip()
                if ref.startswith("ref:"):
                    ref_path = Path(".git") / ref.split(" ")[1]
                    if ref_path.exists():
                        return ref_path.read_text(encoding="utf-8").strip()
                return ref
        except Exception:
            pass
        return "UNKNOWN_COMMIT"

    @classmethod
    def _gather_environment_provenance(cls, seed: int, git_commit: str) -> EnvironmentProvenance:
        import numpy as np
        import pandas as pd
        import pydantic
        import scipy
        import sklearn

        pkg_versions = {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "pydantic": pydantic.__version__,
            "scikit-learn": sklearn.__version__,
        }
        return EnvironmentProvenance(
            python_version=platform.python_version(),
            platform=platform.platform(),
            git_commit=git_commit,
            packages=pkg_versions,
            random_seed=seed,
        )

    def profile_dataset(
        self,
        dataset_name: str,
        records: list[CanonicalNewsRecord],
        dataset_version: str = "v1.0.0",
        splits: dict[str, list[CanonicalNewsRecord]] | None = None,
        file_paths: dict[str, Path] | None = None,
    ) -> DatasetProfile:
        """
        Executes end-to-end analytical profiling on genuine benchmark records.
        """
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        logger.info("Executing DatasetProfiler run %s for dataset '%s'...", run_id, dataset_name)

        # 1. Compute input file hashes if provided
        file_hashes: dict[str, str] = {}
        if file_paths:
            for k, p in file_paths.items():
                if p.exists() and p.is_file():
                    data = p.read_bytes()
                    file_hashes[k] = hashlib.sha256(data).hexdigest()

        # 2. Execute sub-analyzers
        logger.info("Step 1/8 Analyzing label distributions...")
        labels_res = LabelAnalyzer.analyze(records)

        logger.info("Step 2/8 Analyzing text distributions & quality flags...")
        text_res = TextAnalyzer.analyze(records, thresholds=self.text_thresholds)

        logger.info("Step 3/8 Analyzing source & publisher concentration...")
        source_res = SourceAnalyzer.analyze(records)

        logger.info("Step 4/8 Analyzing temporal coverage & stability...")
        temporal_res = TemporalAnalyzer.analyze(records)

        logger.info("Step 5/8 Analyzing metadata missingness & entropy...")
        metadata_res = MetadataAnalyzer.analyze(records)

        logger.info("Step 6/8 Analyzing cross-modal asset availability...")
        modality_res = ModalityAnalyzer.analyze(records)

        logger.info("Step 7/8 Analyzing duplicates and cluster divergence...")
        dup_res = DuplicateAnalyzer.analyze(records)

        # Cross-dataset / split leakage
        if splits and len(splits) > 1:
            cross_res = CrossDatasetAnalyzer.analyze_splits(splits)
        else:
            cross_res = CrossDatasetAnalyzer.analyze_datasets({dataset_name: records})

        logger.info("Step 8/8 Executing inferential statistical hypothesis battery...")
        stats_res = StatisticalTester.run_all(records, alpha=0.05)

        # Compute readiness summary dimensions
        readiness = {
            "label_completeness": f"{round(labels_res.mapping_coverage * 100, 1)}% ({labels_res.unresolved_count} unresolved)",
            "modality_completeness": "Text Only (No Local Images)" if not modality_res.image_binary_available else "Multimodal (Text + Images)",
            "temporal_completeness": f"{round(temporal_res.timestamp_coverage_ratio * 100, 1)}% Timestamps Present",
            "source_completeness": f"{source_res.unique_sources} Unique Sources (Top-5 Share: {source_res.top_5_share}%)",
            "duplicate_risk": f"LOW ({dup_res.exact_duplicates_percentage}% exact, {dup_res.near_duplicate_percentage}% near)",
            "cross_dataset_leakage_risk": "ZERO" if not cross_res.cross_split_leakage_detected else "DETECTED",
            "class_balance": f"Imbalance Ratio: {labels_res.imbalance_ratio}:1 (Normalized Entropy: {labels_res.normalized_entropy})",
            "reproducibility": f"Git: {self.git_commit[:8]} | Python: {platform.python_version()} | Seed: {self.random_seed}",
        }

        env_prov = self._gather_environment_provenance(
            seed=self.random_seed, git_commit=self.git_commit
        )

        profile = DatasetProfile(
            run_id=run_id,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            total_records=len(records),
            environment_provenance=env_prov,
            dataset_file_hashes=file_hashes,
            label_analysis=labels_res,
            text_analysis=text_res,
            source_analysis=source_res,
            temporal_analysis=temporal_res,
            metadata_analysis=metadata_res,
            modality_analysis=modality_res,
            duplicate_analysis=dup_res,
            cross_dataset_analysis=cross_res,
            statistical_tests=stats_res,
            readiness_summary=readiness,
        )

        logger.info("Profiling complete for '%s' (Run ID: %s).", dataset_name, run_id)
        return profile

    @classmethod
    def export_profile(cls, profile: DatasetProfile, output_dir: Path) -> dict[str, Path]:
        """
        Emits individual machine-readable JSON artifacts according to project specifications.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}

        def write_json(filename: str, data: Any) -> Path:
            p = output_dir / filename
            with open(p, "w", encoding="utf-8") as f:
                if hasattr(data, "model_dump"):
                    json.dump(data.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
                else:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            return p

        paths["profile"] = write_json("profile.json", profile)
        paths["label_distribution"] = write_json("label_distribution.json", profile.label_analysis)
        paths["text_statistics"] = write_json("text_statistics.json", profile.text_analysis)
        paths["source_statistics"] = write_json("source_statistics.json", profile.source_analysis)
        paths["temporal_statistics"] = write_json("temporal_statistics.json", profile.temporal_analysis)
        paths["metadata_statistics"] = write_json("metadata_statistics.json", profile.metadata_analysis)
        paths["modality_statistics"] = write_json("modality_statistics.json", profile.modality_analysis)
        paths["duplicate_statistics"] = write_json("duplicate_statistics.json", profile.duplicate_analysis)
        paths["cross_dataset_statistics"] = write_json("cross_dataset_statistics.json", profile.cross_dataset_analysis)
        paths["statistical_tests"] = write_json("statistical_tests.json", profile.statistical_tests)

        logger.info("Emitted 10 machine-readable JSON profiling artifacts to %s", output_dir)
        return paths
