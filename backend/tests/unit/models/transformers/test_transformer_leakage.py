"""Unit tests verifying strict train/validation/test feature and data isolation."""

from pathlib import Path

import torch

from backend.app.models.transformers.dataset import TruthLensClaimDataset
from backend.app.models.transformers.text_classifier import TransformerTextClassifier


def test_partition_identifiers_are_strictly_disjoint():
    """Verify that train, validation, and test partitions have zero overlapping record IDs."""
    data_dir = Path("data/processed/liar")
    train_recs = TruthLensClaimDataset._load_jsonl(data_dir / "train.jsonl")
    val_recs = TruthLensClaimDataset._load_jsonl(data_dir / "val.jsonl")
    test_recs = TruthLensClaimDataset._load_jsonl(data_dir / "test.jsonl")

    train_ids = {r["record_id"] for r in train_recs}
    val_ids = {r["record_id"] for r in val_recs}
    test_ids = {r["record_id"] for r in test_recs}

    # Ensure complete disjointness
    assert len(train_ids & val_ids) == 0, "Leakage detected: Train and Val share record IDs!"
    assert len(train_ids & test_ids) == 0, "Leakage detected: Train and Test share record IDs!"
    assert len(val_ids & test_ids) == 0, "Leakage detected: Val and Test share record IDs!"


def test_training_dataset_strictly_isolates_train_records():
    """Verify that TruthLensClaimDataset for train does not contain validation or test records."""
    data_dir = Path("data/processed/liar")
    val_recs = TruthLensClaimDataset._load_jsonl(data_dir / "val.jsonl")
    test_recs = TruthLensClaimDataset._load_jsonl(data_dir / "test.jsonl")

    val_id_set = {r["record_id"] for r in val_recs}
    test_id_set = {r["record_id"] for r in test_recs}

    classifier = TransformerTextClassifier(
        model_name="bert-base-uncased",
        max_length=32,
        seed=42,
    )

    train_dataset = TruthLensClaimDataset(
        data_dir / "train.jsonl",
        tokenizer=classifier.tokenizer,
        max_length=32,
        split_name="train",
    )

    train_record_ids = set(train_dataset.get_record_ids())
    assert len(train_record_ids & val_id_set) == 0
    assert len(train_record_ids & test_id_set) == 0


def test_validation_evaluation_is_strictly_read_only():
    """Verify that validation evaluation does not update model weights or enable gradients."""
    classifier = TransformerTextClassifier(
        model_name="bert-base-uncased",
        max_length=32,
        seed=42,
    )

    sample_val = [
        {"record_id": "v1", "text": "Validation claim sample A", "normalized_label": "LIKELY_TRUE"},
        {"record_id": "v2", "text": "Validation claim sample B", "normalized_label": "LIKELY_FALSE"},
        {"record_id": "v3", "text": "Validation claim sample C", "normalized_label": "MISLEADING_OR_PARTIALLY_SUPPORTED"},
    ]
    val_dataset = TruthLensClaimDataset(
        sample_val,
        tokenizer=classifier.tokenizer,
        max_length=32,
        split_name="val",
    )
    from torch.utils.data import DataLoader

    val_loader = DataLoader(val_dataset, batch_size=2)

    # Capture weight before evaluation
    param_before = next(classifier.model.parameters()).clone()

    # Run evaluation
    metrics = classifier.evaluate(val_loader)
    assert "eval_loss" in metrics
    assert "overall" in metrics

    # Verify model weights are untouched
    param_after = next(classifier.model.parameters())
    assert torch.equal(param_before, param_after)

    # Verify no gradients were generated
    for p in classifier.model.parameters():
        assert p.grad is None
