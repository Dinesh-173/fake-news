"""Dataset loader for the LIAR benchmark (Wang, ACL 2017)."""

import csv
from pathlib import Path

from backend.app.core.logging import logger
from backend.app.dataset.loaders.base import BaseDatasetLoader
from backend.app.dataset.schemas import CanonicalNewsRecord


class LiarDatasetLoader(BaseDatasetLoader):
    """Loads raw TSV files from the LIAR benchmark into canonical records."""

    def __init__(self):
        super().__init__(dataset_name="liar")

    def load_records(self, source_path: Path) -> list[CanonicalNewsRecord]:
        if not source_path.exists():
            raise FileNotFoundError(f"LIAR dataset file not found at {source_path}")

        records: list[CanonicalNewsRecord] = []
        with open(source_path, encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row_idx, row in enumerate(reader):
                if not row:
                    continue
                # LIAR format specifies 14 columns
                if len(row) < 3:
                    logger.warning("Skipping malformed row %d with fewer than 3 columns", row_idx)
                    continue

                orig_id = row[0].strip() if len(row) > 0 else f"row_{row_idx}"
                orig_label = row[1].strip() if len(row) > 1 else "UNKNOWN"
                statement = row[2].strip() if len(row) > 2 else ""

                # Optional metadata columns in LIAR
                subjects = row[3].strip() if len(row) > 3 else None
                speaker = row[4].strip() if len(row) > 4 else None
                job_title = row[5].strip() if len(row) > 5 else None
                state_info = row[6].strip() if len(row) > 6 else None
                party = row[7].strip() if len(row) > 7 else None
                context = row[13].strip() if len(row) > 13 else None

                metadata = {
                    "subjects": subjects,
                    "speaker": speaker,
                    "job_title": job_title,
                    "state_info": state_info,
                    "party_affiliation": party,
                    "context": context,
                }
                # Credit history counts if present
                if len(row) >= 13:
                    metadata["credit_history"] = {
                        "barely_true": row[8],
                        "false": row[9],
                        "half_true": row[10],
                        "mostly_true": row[11],
                        "pants_on_fire": row[12],
                    }

                record = CanonicalNewsRecord(
                    record_id=self.generate_record_id(orig_id),
                    dataset_name=self.dataset_name,
                    original_id=orig_id,
                    text=statement,
                    title=None,
                    author=speaker,
                    source_name_raw=context or speaker,
                    source_name_normalized=speaker,
                    original_label=orig_label,
                    metadata=metadata,
                )
                records.append(record)

        logger.info("Loaded %d raw records from LIAR dataset %s", len(records), source_path)
        return records
