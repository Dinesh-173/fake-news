"""TruthLens Phase 5 Transformer Experiment Orchestrator.

Manages:
1. Dataset ingestion from authoritative data/processed/liar/ partitions.
2. Pinned model configuration for BERT-Base and RoBERTa-Base.
3. Safe execution modes:
   - 'smoke': Non-research engineering smoke test on small training batches.
   - 'audit': Dataset integrity, hash verification, and configuration audit.
   - 'train': Gated full research training (requires explicit confirmation).
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.models.transformers import (
    INT_TO_LABEL,
    LABEL_TO_INT,
    PINNED_MODELS,
    TransformerModelSerializer,
    TransformerTextClassifier,
    TruthLensClaimDataset,
    calculate_balanced_class_weights,
)

AUTHORITATIVE_HASHES = {
    "train.jsonl": "917e581b69023a116ac01bc2ad9078872ce464a909325105164947a8fbc07ee5",
    "val.jsonl": "406194f5f1a5b12eba94c81ee4fdba269a2242c05c818bf5f702a4c7ab329d56",
    "test.jsonl": "b344c95400cbf91b2fc54fe8e4c06b89506398e7667863bee816d45ca4826330",
}


def compute_file_hash(path: Path) -> str:
    """Calculates SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_dataset_integrity(data_dir: Path) -> dict[str, str]:
    """Verifies that processed dataset partitions match authoritative Phase 4 hashes bit-for-bit."""
    hashes = {}
    for filename, expected_hash in AUTHORITATIVE_HASHES.items():
        filepath = data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Authoritative dataset file not found: {filepath}")
        actual_hash = compute_file_hash(filepath)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Dataset hash mismatch for {filename}!\n"
                f"  Expected: {expected_hash}\n"
                f"  Actual:   {actual_hash}"
            )
        hashes[filename] = actual_hash
    return hashes


def run_smoke_test(
    data_dir: Path,
    model_name: str,
    output_root: Path,
    seed: int = 42,
    num_samples: int = 16,
) -> None:
    """
    Executes a strictly non-research engineering smoke test.
    Validates Dataset, DataLoader, forward pass, loss, backward pass, optimizer step,
    gradient clipping, scheduler, and checkpoint serialization using TRAIN records only.
    """
    print("=" * 70)
    print("TruthLens: Phase 5 Engineering Smoke Test (NON-RESEARCH BENCHMARK)")
    print(f"Model Backbone:    {model_name}")
    print(f"Sample Count:      {num_samples} (TRAIN partition only)")
    print(f"Random Seed:       {seed}")
    print("=" * 70)

    train_path = data_dir / "train.jsonl"
    all_train_records = TruthLensClaimDataset._load_jsonl(train_path)
    smoke_records = all_train_records[:num_samples]

    classifier = TransformerTextClassifier(
        model_name=model_name,
        seed=seed,
        max_length=128,
        learning_rate=2e-5,
    )

    dataset = TruthLensClaimDataset(
        records_or_path=smoke_records,
        tokenizer=classifier.tokenizer,
        max_length=128,
        split_name="smoke_train",
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    classifier.build_optimizer_and_scheduler(total_training_steps=len(dataloader) * 2)

    # Single optimization step
    batch = next(iter(dataloader))
    loss_val = classifier.training_step(batch)
    print(f"[1/3] Optimization Step Success. Loss: {loss_val:.4f}")

    # Forward evaluation
    eval_metrics = classifier.evaluate(dataloader)
    print(f"[2/3] Inference Evaluation Success. Eval Loss: {eval_metrics['eval_loss']:.4f}")

    # Serialization test
    run_dir = TransformerModelSerializer.create_run_directory(
        base_dir=output_root,
        run_id=f"smoke_test_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    ckpt_dir = TransformerModelSerializer.save_checkpoint(
        classifier=classifier,
        output_dir=run_dir,
        checkpoint_name="smoke_checkpoint",
        extra_metadata={"smoke_test": True, "step_loss": loss_val},
    )
    print(f"[3/3] Checkpoint Serialization Success: {ckpt_dir}")

    print("=" * 70)
    print("SMOKE TEST COMPLETE: All infrastructure components operational.")
    print("NOTE: Zero research metrics recorded. Test partition was untouched.")
    print("=" * 70)


def run_research_training(
    data_dir: Path,
    model_name: str,
    candidate: str,
    output_root: Path,
    seed: int = 42,
    epochs: int = 3,
    batch_size: int = 16,
    eval_batch_size: int = 32,
    lr: float = 2e-5,
    max_length: int = 128,
) -> Path:
    """
    Executes a complete, scientifically rigorous research training experiment.
    - Trains strictly on TRAIN data.
    - Validates at each epoch on VAL data; selects best checkpoint via validation Macro-F1.
    - Performs a single, locked evaluation on TEST data using the selected best checkpoint.
    - Records exhaustive experiment telemetry and artifacts.
    """
    print("=" * 75)
    print(f"TruthLens Phase 5 Research Training: {model_name} (Candidate: {candidate.upper()})")
    print(f"Seed: {seed} | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print("=" * 75)

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    test_path = data_dir / "test.jsonl"

    train_records = TruthLensClaimDataset._load_jsonl(train_path)
    val_records = TruthLensClaimDataset._load_jsonl(val_path)

    # Balanced class weights calculation (TRAIN ONLY)
    if candidate == "c3":
        train_labels = [LABEL_TO_INT[r["normalized_label"]] for r in train_records]
        class_weights = calculate_balanced_class_weights(train_labels)
        print(f"Computed training-only balanced class weights: {class_weights.tolist()}")
    else:
        class_weights = None

    effective_lr = 3e-5 if candidate == "c2" else lr
    classifier = TransformerTextClassifier(
        model_name=model_name,
        class_weights=class_weights,
        max_length=max_length,
        learning_rate=effective_lr,
        seed=seed,
    )

    train_dataset = TruthLensClaimDataset(
        train_records,
        tokenizer=classifier.tokenizer,
        max_length=max_length,
        split_name="train",
        pad_to_max_length=False,
    )
    val_dataset = TruthLensClaimDataset(
        val_records,
        tokenizer=classifier.tokenizer,
        max_length=max_length,
        split_name="val",
        pad_to_max_length=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.get_collate_fn(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=val_dataset.get_collate_fn(),
    )

    total_training_steps = len(train_loader) * epochs
    classifier.build_optimizer_and_scheduler(total_training_steps=total_training_steps)

    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    clean_model_id = model_name.replace("/", "_")
    run_id = f"run_{ts_str}_{clean_model_id}_{candidate}"
    run_dir = TransformerModelSerializer.create_run_directory(output_root, run_id)

    best_val_f1 = -1.0
    best_val_recall = -1.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_metrics = None
    epoch_logs = []

    print(f"\nStarting {epochs} epochs of training ({len(train_loader)} steps per epoch)...")
    for epoch in range(1, epochs + 1):
        t_epoch_start = time.perf_counter()
        running_loss = 0.0
        n_samples = 0
        classifier.model.train()

        for step, batch in enumerate(train_loader):
            loss_val = classifier.training_step(batch)
            bs = len(batch["labels"])
            running_loss += loss_val * bs
            n_samples += bs

            if (step + 1) % 150 == 0 or (step + 1) == len(train_loader):
                cur_avg_loss = running_loss / n_samples
                print(f"  Epoch {epoch}/{epochs} | Step {step+1}/{len(train_loader)} | Train Loss: {cur_avg_loss:.4f}")

        avg_train_loss = running_loss / n_samples if n_samples > 0 else 0.0
        val_metrics = classifier.evaluate(val_loader)
        duration_s = time.perf_counter() - t_epoch_start

        val_f1 = val_metrics["overall"]["macro_f1"]
        val_rec = val_metrics["overall"]["macro_recall"]
        val_loss = val_metrics["eval_loss"]

        log_entry = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_macro_f1": round(val_f1, 4),
            "val_macro_recall": round(val_rec, 4),
            "val_accuracy": round(val_metrics["overall"]["accuracy"], 4),
            "duration_seconds": round(duration_s, 2),
            "val_metrics": val_metrics,
        }
        epoch_logs.append(log_entry)
        print(f"--> [Epoch {epoch}] Duration: {duration_s/60:.2f}m | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {val_f1:.4f}")

        # Deterministic tie-breaking: higher Macro-F1 -> higher Macro-Recall -> lower val loss -> earlier epoch
        is_better = False
        if val_f1 > best_val_f1:
            is_better = True
        elif abs(val_f1 - best_val_f1) < 1e-5:
            if val_rec > best_val_recall:
                is_better = True
            elif abs(val_rec - best_val_recall) < 1e-5 and val_loss < best_val_loss:
                is_better = True

        if is_better:
            best_val_f1 = val_f1
            best_val_recall = val_rec
            best_val_loss = val_loss
            best_epoch = epoch
            best_metrics = val_metrics
            TransformerModelSerializer.save_checkpoint(
                classifier=classifier,
                output_dir=run_dir,
                checkpoint_name="best_checkpoint",
                extra_metadata={"best_epoch": best_epoch, "val_metrics": val_metrics},
            )
            print(f"    * New best checkpoint saved (Epoch {epoch}, Val Macro-F1: {val_f1:.4f}) *")

    print(f"\nTraining Complete. Best Checkpoint selected from Epoch {best_epoch} (Val Macro-F1: {best_val_f1:.4f}).")

    # Load best checkpoint for single locked test evaluation
    best_ckpt_dir = run_dir / "checkpoints" / "best_checkpoint"
    classifier.model = AutoModelForSequenceClassification.from_pretrained(best_ckpt_dir)
    classifier.model.to(classifier.device)

    # Single locked test set evaluation
    print("\nExecuting single locked evaluation on TEST set (data/processed/liar/test.jsonl)...")
    test_records = TruthLensClaimDataset._load_jsonl(test_path)
    test_dataset = TruthLensClaimDataset(
        test_records,
        tokenizer=classifier.tokenizer,
        max_length=max_length,
        split_name="test",
        pad_to_max_length=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=test_dataset.get_collate_fn(),
    )

    t_test_start = time.perf_counter()
    test_metrics = classifier.evaluate(test_loader)
    test_duration = time.perf_counter() - t_test_start
    test_latency_ms = (test_duration / len(test_records)) * 1000

    print("=" * 75)
    print("TEST EVALUATION RESULTS (FINAL LOCKED BENCHMARK):")
    print(f"  Test Accuracy:        {test_metrics['overall']['accuracy']:.4f}")
    print(f"  Test Macro-Precision: {test_metrics['overall']['macro_precision']:.4f}")
    print(f"  Test Macro-Recall:    {test_metrics['overall']['macro_recall']:.4f}")
    print(f"  Test Macro-F1:        {test_metrics['overall']['macro_f1']:.4f}")
    print(f"  Inference Latency:    {test_latency_ms:.2f} ms / claim")
    print("=" * 75)

    test_preds, test_probas, test_ids = classifier.predict_proba(test_loader)
    test_predictions_list = [
        {
            "record_id": test_ids[i] if i < len(test_ids) else f"rec_{i}",
            "true_label": test_records[i]["normalized_label"],
            "predicted_label": INT_TO_LABEL[int(test_preds[i])],
            "probabilities": {
                INT_TO_LABEL[c]: round(float(test_probas[i][c]), 5) for c in range(3)
            },
        }
        for i in range(len(test_records))
    ]

    training_summary = {
        "model_name": model_name,
        "candidate": candidate,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "epoch_logs": epoch_logs,
        "test_latency_ms_per_sample": round(test_latency_ms, 2),
        "total_training_duration_minutes": round(sum(e["duration_seconds"] for e in epoch_logs) / 60, 2),
    }

    config_record = {
        "model_name": model_name,
        "candidate": candidate,
        "seed": seed,
        "max_length": max_length,
        "learning_rate": effective_lr,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "epochs": epochs,
        "class_weights": class_weights.tolist() if class_weights is not None else None,
        "tie_break_rule": "higher Macro-F1 > higher Macro-Recall > lower val loss > earlier epoch",
    }

    dataset_snapshot = {
        "data_directory": str(data_dir.resolve()),
        "verified_hashes": AUTHORITATIVE_HASHES,
        "train_count": len(train_records),
        "val_count": len(val_records),
        "test_count": len(test_records),
    }

    TransformerModelSerializer.save_run_artifacts(
        run_dir=run_dir,
        config=config_record,
        dataset_snapshot=dataset_snapshot,
        training_summary=training_summary,
        validation_results=best_metrics,
        test_results=test_metrics,
    )

    with open(run_dir / "predictions_test.json", "w", encoding="utf-8") as f:
        json.dump(test_predictions_list, f, indent=2)

    with open(run_dir / "confusion_matrix_test.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics["confusion_matrix"], f, indent=2)

    print(f"Artifacts and telemetry preserved in: {run_dir}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="TruthLens Phase 5 Transformer Orchestrator.")
    parser.add_argument(
        "--mode",
        choices=["audit", "smoke", "train"],
        default="smoke",
        help="Execution mode: 'audit' (verify assets), 'smoke' (non-research pipeline check), 'train' (research run).",
    )
    parser.add_argument(
        "--model",
        choices=["bert-base-uncased", "roberta-base", "all"],
        default="bert-base-uncased",
        help="Pretrained model architecture.",
    )
    parser.add_argument(
        "--candidate",
        choices=["c1", "c2", "c3"],
        default="c1",
        help="Hyperparameter candidate: c1 (standard CE), c2 (higher LR), c3 (balanced CE).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/liar"),
        help="Directory containing the processed JSONL dataset partitions.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/transformers"),
        help="Root directory for transformer experiment runs.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument(
        "--confirm-research-run",
        action="store_true",
        help="Safety flag required to execute full multi-hour research training.",
    )

    args = parser.parse_args()

    # Step 1: Always verify dataset hashes first
    print("Verifying authoritative dataset hashes...")
    verified_hashes = verify_dataset_integrity(args.data_dir)
    print("Dataset integrity verified: 100% bitwise parity with authoritative Phase 4.")

    if args.mode == "audit":
        print("\n=== PHASE 5 READINESS AUDIT ===")
        print(f"Data Directory:     {args.data_dir.resolve()}")
        print(f"Verified Hashes:    {json.dumps(verified_hashes, indent=2)}")
        print(f"Pinned Models:      {json.dumps(PINNED_MODELS, indent=2)}")
        print(f"Output Root:        {args.output_root.resolve()}")
        env_info = TransformerModelSerializer.get_environment_info()
        print(f"Environment Info:   {json.dumps(env_info, indent=2)}")
        print("Audit Complete.")
        return

    if args.mode == "smoke":
        target_models = (
            ["bert-base-uncased", "roberta-base"] if args.model == "all" else [args.model]
        )
        for m in target_models:
            run_smoke_test(
                data_dir=args.data_dir,
                model_name=m,
                output_root=args.output_root,
                seed=args.seed,
            )
        return

    if args.mode == "train":
        if not args.confirm_research_run:
            print("\n" + "!" * 75)
            print("SAFETY STOP: Full research training requires explicit confirmation.")
            print("Execute with: --confirm-research-run.")
            print("!" * 75)
            sys.exit(1)

        target_models = (
            ["bert-base-uncased", "roberta-base"] if args.model == "all" else [args.model]
        )
        for m in target_models:
            run_research_training(
                data_dir=args.data_dir,
                model_name=m,
                candidate=args.candidate,
                output_root=args.output_root,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                max_length=args.max_length,
            )


if __name__ == "__main__":
    main()
