"""Dataset loader for Fakeddit multimodal benchmark (Nakamura et al., 2020)."""

import csv
from pathlib import Path

from backend.app.core.logging import logger
from backend.app.dataset.loaders.base import BaseDatasetLoader
from backend.app.dataset.schemas import CanonicalNewsRecord


class FakedditLoader(BaseDatasetLoader):
    """Loads Fakeddit multimodal TSV records."""

    def __init__(self):
        super().__init__(dataset_name="fakeddit")

    def load_records(self, source_path: Path) -> list[CanonicalNewsRecord]:
        if not source_path.exists():
            raise FileNotFoundError(f"Fakeddit file not found at {source_path}")

        records: list[CanonicalNewsRecord] = []
        with open(source_path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for idx, row in enumerate(reader):
                orig_id = row.get("id") or f"fakeddit_{idx}"
                text = row.get("clean_title") or row.get("title") or ""
                # 6-way label preferred if available, else 2-way
                orig_label = row.get("6_way_label") or row.get("2_way_label") or "UNKNOWN"
                image_url = row.get("image_url")
                subreddit = row.get("subreddit")

                record = CanonicalNewsRecord(
                    record_id=self.generate_record_id(orig_id),
                    dataset_name=self.dataset_name,
                    original_id=orig_id,
                    text=text,
                    title=row.get("title"),
                    image_url=image_url,
                    source_name_raw=f"r/{subreddit}" if subreddit else None,
                    author=row.get("author"),
                    original_label=str(orig_label).strip(),
                    metadata={
                        "subreddit": subreddit,
                        "domain": row.get("domain"),
                        "has_image": bool(row.get("hasImage") == "1" or image_url),
                    },
                )
                records.append(record)

        logger.info("Loaded %d raw records from Fakeddit dataset %s", len(records), source_path)
        return records
