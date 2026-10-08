"""Native PyTorch Dataset for TruthLens contextual transformer baselines."""

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from backend.app.models.baselines.evaluator import LABEL_TO_INT


class TruthLensClaimDataset(Dataset):
    """
    Native PyTorch Dataset encapsulating canonical TruthLens news claims.

    Loads preprocessed records from JSONL partitions without altering text or partition state.
    Produces tokenized batches with explicit truncation and padding up to max_length.
    """

    def __init__(
        self,
        records_or_path: list[dict[str, Any]] | Path | str,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        split_name: str | None = None,
        pad_to_max_length: bool = True,
    ) -> None:
        """
        Initializes the dataset from a JSONL path or an in-memory list of record dicts.

        Args:
            records_or_path: Path to JSONL partition file or in-memory list of record dicts.
            tokenizer: Pretrained Hugging Face tokenizer instance.
            max_length: Maximum sequence length (tokens). Pinned to 128 for Phase 5.
            split_name: Optional partition descriptor ('train', 'val', 'test') for auditing.
            pad_to_max_length: If True, pads all items to max_length. If False, pads dynamically via get_collate_fn.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split_name = split_name
        self.pad_to_max_length = pad_to_max_length

        if isinstance(records_or_path, (str, Path)):
            self.file_path = Path(records_or_path)
            self.records = self._load_jsonl(self.file_path)
        else:
            self.file_path = None
            self.records = list(records_or_path)

        # Pre-validate labels
        for idx, r in enumerate(self.records):
            lbl = r.get("normalized_label")
            if lbl not in LABEL_TO_INT:
                raise ValueError(
                    f"Record at index {idx} has invalid canonical label: '{lbl}'. "
                    f"Expected one of: {list(LABEL_TO_INT.keys())}"
                )

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        """Reads JSONL records from disk in a strictly read-only manner."""
        if not path.exists():
            raise FileNotFoundError(f"Partition file does not exist: {path}")
        records: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Returns a single tokenized item.

        Returns:
            dict containing:
                - 'input_ids': torch.Tensor of shape (max_length,)
                - 'attention_mask': torch.Tensor of shape (max_length,)
                - 'labels': torch.Tensor scalar (dtype=torch.long)
                - 'record_id': str
                - 'original_id': str
        """
        rec = self.records[idx]
        text = rec.get("text", "")
        label_str = rec["normalized_label"]
        label_id = LABEL_TO_INT[label_str]

        padding_strategy = "max_length" if self.pad_to_max_length else False
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=padding_strategy,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_id, dtype=torch.long),
            "record_id": rec.get("record_id", f"rec_{idx}"),
            "original_id": rec.get("original_id", f"orig_{idx}"),
        }

    def get_collate_fn(self):
        """Returns a collate function for DataLoader that dynamically pads batches up to batch max."""
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0

        def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
            input_ids = [item["input_ids"] for item in batch]
            attention_masks = [item["attention_mask"] for item in batch]
            labels = torch.stack([item["labels"] for item in batch])

            padded_input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=pad_token_id
            )
            padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
                attention_masks, batch_first=True, padding_value=0
            )
            return {
                "input_ids": padded_input_ids,
                "attention_mask": padded_attention_mask,
                "labels": labels,
                "record_id": [item["record_id"] for item in batch],
                "original_id": [item["original_id"] for item in batch],
            }

        return collate_fn

    def get_record_ids(self) -> list[str]:
        """Returns the list of all record_ids in this dataset for audit and leakage verification."""
        return [r.get("record_id", f"rec_{i}") for i, r in enumerate(self.records)]

    def get_labels(self) -> list[int]:
        """Returns integer labels for all records in this dataset."""
        return [LABEL_TO_INT[r["normalized_label"]] for r in self.records]

    def get_label_distribution(self) -> dict[str, int]:
        """Returns distribution of string labels in this dataset."""
        from collections import Counter

        counts = Counter(r["normalized_label"] for r in self.records)
        return dict(counts)
