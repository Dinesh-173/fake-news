"""Label normalizer applying machine-readable mappings to TruthLens credibility states."""

import json
from pathlib import Path

from backend.app.core.logging import logger
from backend.app.dataset.schemas import CanonicalNewsRecord


class LabelNormalizer:
    """Applies verified, dataset-specific label mappings with strict provenance preservation."""

    DEFAULT_MAPPINGS_DIR = (
        Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "label_mappings"
    )

    def __init__(self, mappings_dir: Path | None = None):
        self.mappings_dir = mappings_dir or self.DEFAULT_MAPPINGS_DIR
        self._loaded_mappings: dict[str, dict] = {}

    def load_mapping(self, dataset_name: str) -> dict:
        """Loads and caches label mapping configuration for a dataset."""
        norm_name = dataset_name.lower().replace("-", "_")
        if norm_name in self._loaded_mappings:
            return self._loaded_mappings[norm_name]

        mapping_file = self.mappings_dir / f"{norm_name}.json"
        if not mapping_file.exists():
            logger.warning(
                "No label mapping file found at %s. Unresolved status will be assigned.",
                mapping_file,
            )
            self._loaded_mappings[norm_name] = {}
            return {}

        try:
            with open(mapping_file, encoding="utf-8") as f:
                data = json.load(f)
            self._loaded_mappings[norm_name] = data.get("mappings", {})
            return self._loaded_mappings[norm_name]
        except Exception as exc:
            logger.error("Error reading mapping file %s: %s", mapping_file, exc)
            self._loaded_mappings[norm_name] = {}
            return {}

    def normalize_label(self, dataset_name: str, original_label: str) -> tuple[str | None, str]:
        """
        Maps original label to TruthLens canonical state.
        Returns: (normalized_label, mapping_status)
        where mapping_status is 'mapped', 'unresolved', or 'excluded'.
        """
        if not original_label or not str(original_label).strip():
            return None, "unresolved"

        norm_key = str(original_label).strip().lower()
        mapping = self.load_mapping(dataset_name)

        if norm_key in mapping:
            target = mapping[norm_key].get("target")
            return target, "mapped"

        return None, "unresolved"

    def normalize_record(self, record: CanonicalNewsRecord) -> CanonicalNewsRecord:
        """Normalizes the label for a single canonical record, preserving original_label."""
        target, status = self.normalize_label(record.dataset_name, record.original_label)
        return record.model_copy(
            update={
                "normalized_label": target,
                "label_mapping_status": status,
            }
        )
