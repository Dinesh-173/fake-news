"""CLI runner to execute the full TruthLens dataset pipeline on raw files or test fixtures."""

import argparse
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.dataset.loaders import (
    FakedditLoader,
    FakeNewsNetLoader,
    LiarDatasetLoader,
    NelaGtLoader,
)
from backend.app.dataset.pipeline import DatasetPipeline


def create_synthetic_liar_fixture(file_path: Path) -> Path:
    """Creates a small, deterministic synthetic LIAR TSV fixture strictly for testing the pipeline."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "1001.json\tpants-fire\tThe earth has completely stopped spinning and gravity is failing.\tscience\tJohn Doe\tblogger\tTexas\tindependent\t1\t2\t1\t0\t5\tonline blog",
        "1002.json\tfalse\tUnemployment reached 95 percent across the nation yesterday.\teconomy\tJane Smith\tactivist\tOhio\trepublican\t2\t4\t2\t1\t2\tsocial media post",
        "1003.json\thalf-true\tTax revenues increased by 10 percent following the recent reform.\ttaxes\tBob Senator\tsenator\tCalifornia\tdemocrat\t0\t1\t3\t4\t0\tpress conference",
        "1004.json\tmostly-true\tSolar installations grew substantially over the last three fiscal quarters.\tenergy\tAlice Green\tofficial\tNevada\tdemocrat\t0\t0\t1\t5\t0\tspeech at convention",
        "1005.json\ttrue\tNASA launched the Artemis I mission from Kennedy Space Center.\tspace\tNASA Spokesperson\tspokesperson\tFlorida\tnone\t0\t0\t0\t2\t0\tofficial release",
        "1006.json\tbarely-true\tThe new policy will force all car owners to sell their vehicles.\ttransport\tMark Brown\tpundit\tFlorida\trepublican\t3\t2\t1\t0\t1\tradio interview",
        "1007.json\tfalse\tUnemployment reached 95 percent across the nation yesterday.\teconomy\tJane Smith Copy\tactivist\tOhio\trepublican\t2\t4\t2\t1\t2\twire syndicate",  # Exact duplicate
        "1008.json\ttrue\tNASA launched the Artemis I mission from Kennedy Space Center today.\tspace\tNASA Media\tspokesperson\tFlorida\tnone\t0\t0\t0\t2\t0\tpress room",  # Near duplicate
        "1009.json\tfalse\tWater boils at 20 degrees Celsius under normal sea level conditions.\tscience\tRandom User\tnone\tnone\tnone\t0\t1\t0\t0\t0\tforum comment",
        "1010.json\tmostly-true\tElectric vehicle sales set a quarterly record in the United States.\tauto\tIndustry Analyst\tanalyst\tMichigan\tnone\t0\t0\t1\t3\t0\tindustry report",
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    return file_path


def main():
    parser = argparse.ArgumentParser(description="Run TruthLens Dataset Pipeline.")
    parser.add_argument(
        "--dataset", required=True, choices=["liar", "fakenewsnet", "fakeddit", "nela"]
    )
    parser.add_argument("--input", type=Path, help="Path to raw dataset file.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/test_run"))
    parser.add_argument("--version", default="v1.0.0")
    parser.add_argument(
        "--fixture", action="store_true", help="Generate and run on synthetic test fixture."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategy", choices=["stratified", "temporal", "random"], default="stratified"
    )

    args = parser.parse_args()

    loaders = {
        "liar": LiarDatasetLoader,
        "fakenewsnet": FakeNewsNetLoader,
        "fakeddit": FakedditLoader,
        "nela": NelaGtLoader,
    }

    loader = loaders[args.dataset]()

    if args.fixture:
        fixture_path = Path("data/raw/fixtures/synthetic_liar_sample.tsv")
        create_synthetic_liar_fixture(fixture_path)
        input_path = fixture_path
        source_origin = "TruthLens Synthetic Test Fixture"
        terms = "MIT License (Test Fixture Only)"
    else:
        if not args.input:
            print("ERROR: --input path is required when not using --fixture.")
            sys.exit(1)
        input_path = args.input
        source_origin = f"Dataset File: {input_path.name}"
        terms = "Academic Research Use"

    pipeline = DatasetPipeline(loader=loader, random_seed=args.seed)
    records, manifest, report_path = pipeline.run_pipeline(
        source_path=input_path,
        output_dir=args.output,
        dataset_version=args.version,
        source_url_or_origin=source_origin,
        license_or_terms=terms,
        split_strategy=args.strategy,
    )

    print("=" * 60)
    print("TruthLens Dataset Pipeline Execution Complete")
    print(f"Dataset:       {manifest.dataset_name} ({manifest.dataset_version})")
    print(f"Total Records: {len(records):,}")
    print(
        f"Splits:        Train={manifest.split_statistics.train_count}, Val={manifest.split_statistics.val_count}, Test={manifest.split_statistics.test_count}"
    )
    print(
        f"Duplicates:    {manifest.duplicate_statistics.exact_duplicates_count} exact, {manifest.duplicate_statistics.near_duplicate_groups} near clusters"
    )
    print(f"Manifest:      {args.output / 'manifest.json'}")
    print(f"Report:        {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
