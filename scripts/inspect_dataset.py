"""Dataset inspection utility for analyzing raw format, encoding, and row samples."""

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


def inspect_dataset(dataset_name: str, file_path: Path, max_samples: int = 3):
    print("=" * 60)
    print(f"TruthLens Dataset Inspector: {dataset_name.upper()}")
    print(f"Target Path: {file_path}")
    print("=" * 60)

    if not file_path.exists():
        print(f"ERROR: Target file does not exist at {file_path}")
        return

    loaders = {
        "liar": LiarDatasetLoader,
        "fakenewsnet": FakeNewsNetLoader,
        "fakeddit": FakedditLoader,
        "nela": NelaGtLoader,
    }

    loader_cls = loaders.get(dataset_name.lower())
    if not loader_cls:
        print(f"ERROR: Unsupported dataset '{dataset_name}'. Supported: {list(loaders.keys())}")
        return

    loader = loader_cls()
    try:
        records = loader.load_records(file_path)
        print(f"Total raw records loaded: {len(records):,}")
        print("-" * 60)
        print(f"Showing first {min(max_samples, len(records))} sample records:")
        for idx, rec in enumerate(records[:max_samples]):
            print(f"\n[Sample {idx + 1}]")
            print(f"  Record ID:      {rec.record_id}")
            print(f"  Original ID:    {rec.original_id}")
            print(f"  Original Label: {rec.original_label}")
            print(f"  Title:          {rec.title or '(None)'}")
            print(f"  Text Snippet:   {rec.text[:120]}...")
            print(f"  Author/Speaker: {rec.author or '(None)'}")
            print(f"  Source:         {rec.source_name_raw or '(None)'}")
    except Exception as exc:
        print(f"ERROR loading records: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect raw dataset files.")
    parser.add_argument(
        "--dataset", required=True, choices=["liar", "fakenewsnet", "fakeddit", "nela"]
    )
    parser.add_argument("--file", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()

    inspect_dataset(args.dataset, args.file, args.samples)
