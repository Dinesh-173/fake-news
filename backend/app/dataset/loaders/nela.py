"""Dataset loader for NELA-GT news corpus (Gruppi et al., 2018-2022)."""

import json
from pathlib import Path

from backend.app.core.logging import logger
from backend.app.dataset.loaders.base import BaseDatasetLoader
from backend.app.dataset.schemas import CanonicalNewsRecord


class NelaGtLoader(BaseDatasetLoader):
    """Loads NELA-GT articles from JSONL or JSON formats."""

    def __init__(self):
        super().__init__(dataset_name="nela_gt")

    def load_records(self, source_path: Path) -> list[CanonicalNewsRecord]:
        if not source_path.exists():
            raise FileNotFoundError(f"NELA-GT file not found at {source_path}")

        records: list[CanonicalNewsRecord] = []
        with open(source_path, encoding="utf-8", errors="replace") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                orig_id = str(item.get("id", f"nela_{line_idx}"))
                text = item.get("content") or item.get("text") or ""
                title = item.get("title")
                source = item.get("source")
                url = item.get("url")
                date_str = item.get("date") or item.get("published")
                label = item.get("label") or item.get("reliability") or "unreliable"

                record = CanonicalNewsRecord(
                    record_id=self.generate_record_id(orig_id),
                    dataset_name=self.dataset_name,
                    original_id=orig_id,
                    text=text,
                    title=title,
                    source_url=url,
                    source_name_raw=source,
                    published_at_raw=date_str,
                    original_label=str(label).strip().lower(),
                    metadata={"author": item.get("author")},
                )
                records.append(record)

        logger.info("Loaded %d raw records from NELA-GT dataset %s", len(records), source_path)
        return records
