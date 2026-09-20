"""Abstract base class for dataset loaders with deterministic record identification."""

import abc
import uuid
from pathlib import Path

from backend.app.dataset.schemas import CanonicalNewsRecord


class BaseDatasetLoader(abc.ABC):
    """Abstract dataset loader interface."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def generate_record_id(self, original_id: str) -> str:
        """Generates a deterministic UUID based on dataset name and original identifier."""
        clean_id = str(original_id).strip()
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.dataset_name}:{clean_id}"))

    @abc.abstractmethod
    def load_records(self, source_path: Path) -> list[CanonicalNewsRecord]:
        """Loads raw dataset file and converts into raw canonical records."""
        raise NotImplementedError
