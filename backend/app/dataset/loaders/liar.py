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

    def _load_single_tsv(
        self, file_path: Path, default_split: str | None = None
    ) -> list[CanonicalNewsRecord]:
        records: list[CanonicalNewsRecord] = []
        with open(file_path, encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row_idx, row in enumerate(reader):
                if not row:
                    continue
                # LIAR format specifies 14 columns
                if len(row) < 3:
                    logger.warning(
                        "Skipping malformed row %d with fewer than 3 columns in %s",
                        row_idx,
                        file_path,
                    )
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
                    split=default_split,
                    original_split=default_split,
                )
                records.append(record)
        return records

    def load_records(self, source_path: Path) -> list[CanonicalNewsRecord]:
        if not source_path.exists():
            raise FileNotFoundError(f"LIAR dataset file or directory not found at {source_path}")

        if source_path.is_dir():
            train_p = source_path / "train.tsv"
            valid_p = source_path / "valid.tsv"
            if not valid_p.exists() and (source_path / "val.tsv").exists():
                valid_p = source_path / "val.tsv"
            test_p = source_path / "test.tsv"

            if not (train_p.exists() and valid_p.exists() and test_p.exists()):
                raise FileNotFoundError(
                    f"Directory {source_path} is missing required LIAR benchmark TSV files (train.tsv, valid.tsv, test.tsv)"
                )

            train_records = self._load_single_tsv(train_p, default_split="train")
            val_records = self._load_single_tsv(valid_p, default_split="val")
            test_records = self._load_single_tsv(test_p, default_split="test")

            total_records = len(train_records) + len(val_records) + len(test_records)
            logger.info(
                "Loaded authentic LIAR benchmark from %s: train=%d, val=%d, test=%d, total=%d",
                source_path,
                len(train_records),
                len(val_records),
                len(test_records),
                total_records,
            )
            return train_records + val_records + test_records

        # Single file fallback
        fname = source_path.name.lower()
        split = None
        if "train" in fname:
            split = "train"
        elif "val" in fname or "valid" in fname:
            split = "val"
        elif "test" in fname:
            split = "test"

        records = self._load_single_tsv(source_path, default_split=split)
        logger.info("Loaded %d raw records from LIAR dataset %s", len(records), source_path)
        return records
