"""Dataset acquisition script for downloading and extracting verified academic benchmarks."""

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.core.logging import logger

DATASET_CATALOG = {
    "liar": {
        "name": "LIAR: A Benchmark Dataset for Fake News Detection",
        "author": "William Yang Wang (ACL 2017)",
        "source_url": "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip",
        "expected_files": ["train.tsv", "valid.tsv", "test.tsv", "README"],
        "license": "Creative Commons Attribution 4.0 International / Academic Research",
        "archive_name": "liar_dataset.zip",
    },
    "fakenewsnet": {
        "name": "FakeNewsNet: A Data Repository with News Content, Social Context, and Spatiotemporal Information",
        "author": "Kai Shu et al. (ACM SIGKDD 2019)",
        "source_url": "https://github.com/KaiDMML/FakeNewsNet",
        "license": "Academic Research Use Only",
        "acquisition_note": "Requires Twitter API keys or historical tweet hydration for social context.",
    },
    "fakeddit": {
        "name": "Fakeddit: A Multi-Modal Dataset for Fine-Grained Fake News Detection",
        "author": "Kai Nakamura et al. (LREC 2020)",
        "source_url": "https://github.com/entitize/Fakeddit",
        "license": "Academic Research Use Only",
        "acquisition_note": "Requires Reddit image download script or direct Google Drive academic access.",
    },
    "nela": {
        "name": "NELA-GT: News and Factuality Benchmark",
        "author": "Mauricio Gruppi et al. (Dataverse, 2018-2022)",
        "source_url": "https://dataverse.harvard.edu/dataverse/nela",
        "license": "Harvard Dataverse Terms of Use",
        "acquisition_note": "Available via Harvard Dataverse persistent identifier.",
    },
}


def download_and_extract_liar(target_dir: Path) -> bool:
    """Downloads and extracts the official LIAR benchmark from UCSB."""
    info = DATASET_CATALOG["liar"]
    url = info["source_url"]
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / info["archive_name"]

    logger.info("Downloading official LIAR dataset from %s...", url)
    print(f"Downloading {info['name']} from {url}...")

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "TruthLens Research Prototype (Academic Benchmarking)"},
        )
        with urllib.request.urlopen(req, timeout=30) as response, open(zip_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)

        # Compute archive hash
        sha256 = hashlib.sha256(data).hexdigest()
        print(f"Downloaded archive successfully. Size: {len(data):,} bytes. SHA-256: {sha256}")

        # Extract archive
        print("Extracting TSV files...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)

        # Remove archive to save disk space
        zip_path.unlink()

        for expected in info["expected_files"]:
            extracted_file = target_dir / expected
            if extracted_file.exists():
                print(f"  [OK] Extracted: {expected} ({extracted_file.stat().st_size:,} bytes)")
            else:
                print(f"  [WARN] Expected file not found: {expected}")

        return True
    except Exception as exc:
        print(f"ERROR downloading LIAR dataset: {exc}")
        return False


def report_acquisition_status():
    """Prints acquisition and local availability status for all catalogued benchmarks."""
    raw_dir = Path("data/raw")
    print("=" * 70)
    print("TruthLens Benchmark Acquisition & Availability Status")
    print("=" * 70)

    for key, info in DATASET_CATALOG.items():
        ds_dir = raw_dir / key
        is_available = ds_dir.exists() and any(ds_dir.glob("*.tsv")) or any(ds_dir.glob("*.csv")) or any(ds_dir.glob("*.json"))
        print(f"\nDataset: {info['name']}")
        print(f"  Identifier:        {key}")
        print(f"  Author/Source:     {info['author']}")
        print(f"  License:           {info['license']}")
        print(f"  Source URL:        {info['source_url']}")
        if is_available:
            files = [f.name for f in ds_dir.iterdir() if f.is_file()]
            print("  Status:            AVAILABLE LOCALLY")
            print(f"  Local Path:        {ds_dir}")
            print(f"  Files:             {files}")
        else:
            print("  Status:            NOT AVAILABLE LOCALLY")
            if "acquisition_note" in info:
                print(f"  Acquisition Note:  {info['acquisition_note']}")
            else:
                print(f"  Acquisition Cmd:   python scripts/acquire_datasets.py --dataset {key}")
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Acquire and inspect academic datasets.")
    parser.add_argument("--dataset", choices=["liar", "status"], default="status")
    parser.add_argument("--output", type=Path, default=Path("data/raw/liar"))
    args = parser.parse_args()

    if args.dataset == "status":
        report_acquisition_status()
    elif args.dataset == "liar":
        success = download_and_extract_liar(args.output)
        if success:
            print("\nLIAR benchmark acquisition successfully completed.")
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
