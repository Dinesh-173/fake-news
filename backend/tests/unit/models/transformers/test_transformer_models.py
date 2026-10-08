"""Unit tests for Phase 5 transformer model wrapper and loss configurations."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.app.models.transformers.dataset import TruthLensClaimDataset
from backend.app.models.transformers.text_classifier import (
    PINNED_MODELS,
    TransformerTextClassifier,
    calculate_balanced_class_weights,
)


def test_pinned_model_definitions():
    """Verify pinned model registry contains immutable commit SHAs for both architectures."""
    assert "bert-base-uncased" in PINNED_MODELS
    assert "roberta-base" in PINNED_MODELS

    bert_meta = PINNED_MODELS["bert-base-uncased"]
    assert bert_meta["repo_id"] == "google-bert/bert-base-uncased"
    assert bert_meta["revision"] == "86b5e0934494bd15c9632b12f734a8a67f723594"

    roberta_meta = PINNED_MODELS["roberta-base"]
    assert roberta_meta["repo_id"] == "FacebookAI/roberta-base"
    assert roberta_meta["revision"] == "e2da8e2f811d1448a5b465c236feacd80ffbac7b"


def test_balanced_class_weights_calculation():
    """Verify balanced class weights formula: weight_c = N / (K * N_c)."""
    # Synthetic labels with known distribution
    # Class 0: 2, Class 1: 3, Class 2: 5 -> Total N = 10, K = 3
    labels = [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]
    weights = calculate_balanced_class_weights(labels)

    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (3,)
    assert abs(weights[0].item() - (10.0 / (3.0 * 2.0))) < 1e-5
    assert abs(weights[1].item() - (10.0 / (3.0 * 3.0))) < 1e-5
    assert abs(weights[2].item() - (10.0 / (3.0 * 5.0))) < 1e-5


def test_transformer_classifier_forward_and_logits_shape():
    """Verify model produces logits of shape [batch_size, 3] and num_labels=3."""
    classifier = TransformerTextClassifier(
        model_name="bert-base-uncased",
        num_labels=3,
        max_length=64,
        seed=42,
    )

    assert classifier.model.num_labels == 3

    sample_records = [
        {"record_id": "r1", "text": "Claim statement A", "normalized_label": "LIKELY_TRUE"},
        {"record_id": "r2", "text": "Claim statement B", "normalized_label": "LIKELY_FALSE"},
    ]
    dataset = TruthLensClaimDataset(
        sample_records,
        tokenizer=classifier.tokenizer,
        max_length=64,
    )
    dataloader = DataLoader(dataset, batch_size=2)
    batch = next(iter(dataloader))

    with torch.no_grad():
        outputs = classifier.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

    assert outputs.logits.shape == (2, 3)


def test_class_weights_affect_loss():
    """Verify that class-weighted CrossEntropyLoss computes a different loss than unweighted CE."""
    logits = torch.tensor([[2.0, -1.0, 0.5], [-0.5, 1.5, 0.0]], dtype=torch.float32)
    labels = torch.tensor([0, 2], dtype=torch.long)

    # Standard unweighted loss
    loss_unweighted = nn.CrossEntropyLoss()(logits, labels)

    # Class-weighted loss
    weights = torch.tensor([2.5, 0.5, 1.0], dtype=torch.float32)
    loss_weighted = nn.CrossEntropyLoss(weight=weights)(logits, labels)

    assert loss_unweighted.item() != loss_weighted.item()


def test_gradient_clipping_and_scheduler_configuration():
    """Verify gradient clipping threshold and scheduler step initialization."""
    classifier = TransformerTextClassifier(
        model_name="bert-base-uncased",
        max_grad_norm=1.0,
        warmup_ratio=0.10,
        seed=42,
    )
    assert classifier.max_grad_norm == 1.0

    total_steps = 100
    classifier.build_optimizer_and_scheduler(total_training_steps=total_steps)

    assert classifier.optimizer is not None
    assert classifier.scheduler is not None
    # Check initial learning rate
    lr = classifier.scheduler.get_last_lr()[0]
    assert lr >= 0.0
