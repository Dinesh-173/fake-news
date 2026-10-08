"""Serialization and artifact management for Phase 5 transformer experiments."""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
import yaml


def get_git_commit() -> str:
    """Retrieves current Git commit SHA."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


class TransformerModelSerializer:
    """Manages experiment directory creation, telemetry logging, and model checkpoints."""

    @classmethod
    def get_environment_info(cls) -> dict[str, Any]:
        """Collects exhaustive runtime environment information."""
        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "git_commit": get_git_commit(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def create_run_directory(
        cls, base_dir: Path | str = "experiments/transformers", run_id: str | None = None
    ) -> Path:
        """Creates a timestamped experiment run directory."""
        base = Path(base_dir)
        if run_id is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{ts}_transformers"
        run_dir = base / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @classmethod
    def save_checkpoint(
        cls,
        classifier: Any,
        output_dir: Path,
        checkpoint_name: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> Path:
        """
        Saves a model checkpoint using Hugging Face save_pretrained.
        Creates companion metadata JSON for complete experiment provenance.
        """
        ckpt_dir = output_dir / "checkpoints" / checkpoint_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save weights and tokenizer
        classifier.model.save_pretrained(ckpt_dir)
        classifier.tokenizer.save_pretrained(ckpt_dir)

        # Write lightweight companion metadata JSON
        meta = classifier.get_metadata()
        meta["saved_at"] = datetime.now(timezone.utc).isoformat()
        meta["checkpoint_name"] = checkpoint_name
        if extra_metadata:
            meta["extra"] = extra_metadata

        meta_file = ckpt_dir / f"{checkpoint_name}_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return ckpt_dir

    @classmethod
    def save_run_artifacts(
        cls,
        run_dir: Path,
        config: dict[str, Any],
        dataset_snapshot: dict[str, Any],
        training_summary: dict[str, Any] | None = None,
        validation_results: dict[str, Any] | None = None,
        test_results: dict[str, Any] | None = None,
        model_comparison: list[dict[str, Any]] | None = None,
    ) -> None:
        """Writes all structured experiment artifacts in standard JSON/YAML formats."""
        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. config.yaml
        config_path = run_dir / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        # 2. environment.json
        env_path = run_dir / "environment.json"
        with open(env_path, "w", encoding="utf-8") as f:
            json.dump(cls.get_environment_info(), f, indent=2)

        # 3. dataset_snapshot.json
        ds_path = run_dir / "dataset_snapshot.json"
        with open(ds_path, "w", encoding="utf-8") as f:
            json.dump(dataset_snapshot, f, indent=2)

        # 4. training_summary.json
        if training_summary:
            ts_path = run_dir / "training_summary.json"
            with open(ts_path, "w", encoding="utf-8") as f:
                json.dump(training_summary, f, indent=2)

        # 5. validation_results.json
        if validation_results:
            val_path = run_dir / "validation_results.json"
            with open(val_path, "w", encoding="utf-8") as f:
                json.dump(validation_results, f, indent=2)

        # 6. test_results.json
        if test_results:
            test_path = run_dir / "test_results.json"
            with open(test_path, "w", encoding="utf-8") as f:
                json.dump(test_results, f, indent=2)

        # 7. model_comparison.json
        if model_comparison:
            comp_path = run_dir / "model_comparison.json"
            with open(comp_path, "w", encoding="utf-8") as f:
                json.dump(model_comparison, f, indent=2)
