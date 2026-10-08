"""Unit tests verifying deterministic reproducibility across identical transformer configurations."""

import torch
from torch.utils.data import DataLoader

from backend.app.models.transformers.dataset import TruthLensClaimDataset
from backend.app.models.transformers.text_classifier import TransformerTextClassifier


def test_transformer_reproducible_evaluation():
    """Verify two separately instantiated models with identical seed produce identical logits and predictions."""
    sample_records = [
        {"record_id": "r1", "text": "Unemployment rate fell in several key regions.", "normalized_label": "LIKELY_TRUE"},
        {"record_id": "r2", "text": "A secret treaty was signed last night in private.", "normalized_label": "LIKELY_FALSE"},
        {"record_id": "r3", "text": "Official budget surplus reports were released.", "normalized_label": "MISLEADING_OR_PARTIALLY_SUPPORTED"},
    ]

    # Model 1
    m1 = TransformerTextClassifier(
        model_name="bert-base-uncased",
        max_length=32,
        seed=42,
    )
    ds1 = TruthLensClaimDataset(sample_records, tokenizer=m1.tokenizer, max_length=32)
    dl1 = DataLoader(ds1, batch_size=3, shuffle=False)
    preds1, probas1, _ = m1.predict_proba(dl1)

    # Model 2
    m2 = TransformerTextClassifier(
        model_name="bert-base-uncased",
        max_length=32,
        seed=42,
    )
    ds2 = TruthLensClaimDataset(sample_records, tokenizer=m2.tokenizer, max_length=32)
    dl2 = DataLoader(ds2, batch_size=3, shuffle=False)
    preds2, probas2, _ = m2.predict_proba(dl2)

    # Assert exact match of predictions and probabilities
    assert preds1.tolist() == preds2.tolist()
    assert torch.allclose(torch.tensor(probas1), torch.tensor(probas2), atol=1e-5)
