"""Dataset loader for FakeNewsNet (Shu et al., ACM SIGKDD 2019)."""

import csv
import json
from pathlib import Path

from backend.app.core.logging import logger
from backend.app.dataset.loaders.base import BaseDatasetLoader
from backend.app.dataset.schemas import CanonicalNewsRecord


class FakeNewsNetLoader(BaseDatasetLoader):
    """Loads FakeNewsNet CSV/JSON records into canonical format."""

    def __init__(self):
        super().__init__(dataset_name="fakenewsnet")

    def load_records(self, source_path: Path) -> list[CanonicalNewsRecord]:
        if not source_path.exists():
            raise FileNotFoundError(f"FakeNewsNet file not found at {source_path}")

        records: list[CanonicalNewsRecord] = []

        # Support CSV or JSON format
        if source_path.suffix.lower() == ".json":
            with open(source_path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    data = list(data.values())
                for idx, item in enumerate(data):
                    orig_id = str(item.get("id", f"fn_{idx}"))
                    record = CanonicalNewsRecord(
                        record_id=self.generate_record_id(orig_id),
                        dataset_name=self.dataset_name,
                        original_id=orig_id,
                        text=item.get("text") or item.get("title") or "",
                        title=item.get("title"),
                        source_url=item.get("news_url") or item.get("url"),
                        source_name_raw=item.get("source"),
                        image_url=item.get("top_img") or item.get("image_url"),
                        original_label=str(item.get("label", "UNKNOWN")).lower(),
                        metadata={
                            "authors": item.get("authors", []),
                            "keywords": item.get("keywords", []),
                        },
                    )
                    records.append(record)
        else:
            # CSV format: id,news_url,title,tweet_ids
            # Label is often derived from filename (e.g. politifact_fake.csv -> fake)
            default_label = "fake" if "fake" in source_path.name.lower() else "real"
            with open(source_path, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    orig_id = row.get("id") or f"fnn_{idx}"
                    text = row.get("title") or row.get("text") or ""
                    record = CanonicalNewsRecord(
                        record_id=self.generate_record_id(orig_id),
                        dataset_name=self.dataset_name,
                        original_id=orig_id,
                        text=text,
                        title=row.get("title"),
                        source_url=row.get("news_url"),
                        original_label=row.get("label") or default_label,
                        metadata={"tweet_ids": row.get("tweet_ids")},
                    )
                    records.append(record)

        logger.info("Loaded %d raw records from FakeNewsNet dataset %s", len(records), source_path)
        return records
