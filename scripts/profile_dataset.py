"""CLI runner to execute full statistical profiling and quality reporting on genuine datasets."""

import argparse
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.logging import logger
from backend.app.dataset.analysis.profiler import DatasetProfiler
from backend.app.dataset.analysis.report import QualityReportGenerator
from backend.app.dataset.cleaning.cleaner import DataCleaner
from backend.app.dataset.deduplication.deduplicator import Deduplicator
from backend.app.dataset.loaders.liar import LiarDatasetLoader
from backend.app.dataset.normalization.labels import LabelNormalizer
from backend.app.dataset.normalization.sources import SourceNormalizer
from backend.app.dataset.schemas import CanonicalNewsRecord
from backend.app.dataset.validation.validator import DataValidator


def load_and_preprocess_liar(
    liar_dir: Path,
    include_splits: list[str] | None = None,
) -> tuple[list[CanonicalNewsRecord], dict[str, list[CanonicalNewsRecord]], dict[str, Path]]:
    """
    Loads authentic LIAR benchmark TSV files and executes standardized cleaning,
    deduplication, and label normalization.
    """
    splits_to_load = include_splits or ["train", "valid", "test"]
    loader = LiarDatasetLoader()
    label_norm = LabelNormalizer()

    all_raw: list[CanonicalNewsRecord] = []
    split_map_raw: dict[str, list[CanonicalNewsRecord]] = {}
    file_paths: dict[str, Path] = {}

    for s in splits_to_load:
        fname = f"{s}.tsv" if s != "val" else "valid.tsv"
        p = liar_dir / fname
        if not p.exists():
            # Try alternate naming
            if s == "val" and (liar_dir / "val.tsv").exists():
                p = liar_dir / "val.tsv"
            elif s == "valid" and (liar_dir / "valid.tsv").exists():
                p = liar_dir / "valid.tsv"

        if p.exists():
            file_paths[s] = p
            recs = loader.load_records(p)
            for r in recs:
                r_split = r.model_copy(update={"split": s})
                all_raw.append(r_split)
            split_map_raw[s] = recs
            logger.info("Loaded %d authentic records from %s", len(recs), p)
        else:
            logger.warning("Split file %s not found in %s", fname, liar_dir)

    if not all_raw:
        raise FileNotFoundError(f"No valid TSV records found in {liar_dir}")

    # Standardized validation and cleaning
    validated, _ = DataValidator.validate_dataset(all_raw)
    cleaned = [DataCleaner.clean_record(r) for r in validated]

    # Deduplication & Cluster assignment
    deduped, _ = Deduplicator.process_duplicates(cleaned, near_duplicate_threshold=0.85)

    # Label & Source Normalization
    normalized: list[CanonicalNewsRecord] = []
    split_map_final: dict[str, list[CanonicalNewsRecord]] = {s: [] for s in split_map_raw}

    for r in deduped:
        r_norm = label_norm.normalize_record(r)
        domain = SourceNormalizer.extract_domain(r_norm.source_url or r_norm.source_name_raw)
        norm_src = SourceNormalizer.normalize_source_name(r_norm.source_name_raw)
        final_r = r_norm.model_copy(
            update={
                "source_domain": domain,
                "source_name_normalized": norm_src or r_norm.source_name_normalized,
            }
        )
        normalized.append(final_r)
        if final_r.split and final_r.split in split_map_final:
            split_map_final[final_r.split].append(final_r)

    return normalized, split_map_final, file_paths


def main():
    parser = argparse.ArgumentParser(description="Profile real benchmark datasets for TruthLens.")
    parser.add_argument(
        "--dataset",
        choices=["liar", "fakenewsnet", "fakeddit", "nela"],
        default="liar",
        help="Dataset benchmark identifier to profile.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/liar"),
        help="Directory containing the raw benchmark files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/eda"),
        help="Root directory where run artifacts are saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    if args.dataset == "liar":
        if not args.data_dir.exists():
            print(f"ERROR: LIAR data directory not found at {args.data_dir}.")
            print("Please acquire it first: python scripts/acquire_datasets.py --dataset liar")
            sys.exit(1)

        print(f"Loading and validating genuine {args.dataset.upper()} benchmark files from {args.data_dir}...")
        records, split_map, file_paths = load_and_preprocess_liar(args.data_dir)
        print(f"Loaded {len(records):,} total authentic records across {len(split_map)} splits.")

        profiler = DatasetProfiler(random_seed=args.seed)
        profile = profiler.profile_dataset(
            dataset_name=args.dataset,
            records=records,
            dataset_version="v1.0.0 (ACL 2017)",
            splits=split_map,
            file_paths=file_paths,
        )

        run_output_dir = args.output_root / profile.run_id
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export machine-readable JSON artifacts
        json_paths = profiler.export_profile(profile, run_output_dir)

        # 2. Export human-readable Markdown quality report
        report_path = run_output_dir / "dataset_quality_report.md"
        QualityReportGenerator.save_report(profile, report_path)

        print("\n" + "=" * 70)
        print(f"TruthLens Phase 3 Profiling Complete: {profile.dataset_name.upper()}")
        print(f"Run ID:            {profile.run_id}")
        print(f"Records Profiled:  {profile.total_records:,}")
        print(f"Output Directory:  {run_output_dir}")
        print(f"Markdown Report:   {report_path}")
        print(f"JSON Artifacts:    {len(json_paths)} files emitted")
        print("=" * 70)
        print("\nTo generate publication EDA vector figures, execute:")
        print(f"python scripts/generate_eda.py --run-dir {run_output_dir}\n")

    else:
        print("STATUS: NOT AVAILABLE")
        print(f"REASON: dataset '{args.dataset}' not locally acquired in data/raw/{args.dataset}")
        print("ACTION REQUIRED: inspect acquisition notes in scripts/acquire_datasets.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
