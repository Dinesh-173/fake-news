"""TruthLens Phase 5 Transformer Models Package.

Provides native PyTorch Dataset, tokenization, model wrapping, and serialization
for contextual transformer text baselines (BERT and RoBERTa) on the LIAR benchmark.
"""

from backend.app.models.baselines.evaluator import (
    CANONICAL_LABELS,
    INT_TO_LABEL,
    LABEL_TO_INT,
    BaselineEvaluator,
)
from backend.app.models.transformers.dataset import TruthLensClaimDataset
from backend.app.models.transformers.serialization import TransformerModelSerializer
from backend.app.models.transformers.text_classifier import (
    PINNED_MODELS,
    TransformerTextClassifier,
    calculate_balanced_class_weights,
)

__all__ = [
    "CANONICAL_LABELS",
    "INT_TO_LABEL",
    "LABEL_TO_INT",
    "PINNED_MODELS",
    "BaselineEvaluator",
    "TruthLensClaimDataset",
    "TransformerModelSerializer",
    "TransformerTextClassifier",
    "calculate_balanced_class_weights",
]
