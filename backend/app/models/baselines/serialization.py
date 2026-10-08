"""Safe serialization and metadata recording for baseline models."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn


class ModelSerializer:
    """Handles serialization of model weights and reproducibility metadata."""

    @classmethod
    def save_model(
        cls,
        model: Any,
        output_dir: Path,
        model_name: str,
        experiment_id: str,
        dataset_manifest_hash: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        """
        Saves trained model using joblib and records a companion metadata JSON with full provenance.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        model_file = output_dir / f"{model_name}.joblib"
        meta_file = output_dir / f"{model_name}_metadata.json"

        # Save model state
        joblib.dump(model, model_file)

        # Build provenance metadata
        metadata: dict[str, Any] = {
            "model_name": model_name,
            "experiment_id": experiment_id,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "dataset_manifest_hash": dataset_manifest_hash,
            "model_params": model.get_params() if hasattr(model, "get_params") else {},
        }
        if extra_metadata:
            metadata["extra"] = extra_metadata

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return model_file, meta_file

    @classmethod
    def load_model(cls, model_file: Path) -> Any:
        """Loads a saved model from disk."""
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        return joblib.load(model_file)
