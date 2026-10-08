"""Baseline A: Majority Class Classifier serving as empirical sanity floor."""

from collections import Counter
from typing import Any

import numpy as np

from backend.app.models.baselines.evaluator import CANONICAL_LABELS, LABEL_TO_INT


class MajorityClassifier:
    """
    Trivial reference baseline predicting the empirical majority class from training data.
    Establishes the performance floor derived strictly from class distribution.
    """

    def __init__(self):
        self.majority_label: str | None = None
        self.majority_index: int | None = None
        self.class_counts: dict[str, int] = {}
        self.class_priors: dict[str, float] = {}
        self.prior_distribution: np.ndarray | None = None
        self.is_fitted: bool = False

    def fit(self, X: list[str] | np.ndarray, y: list[str] | np.ndarray) -> "MajorityClassifier":
        """Fits the majority classifier on training labels."""
        if len(y) == 0:
            raise ValueError("Cannot fit MajorityClassifier on empty training targets.")

        y_labels = [str(lbl) for lbl in y]
        counts = Counter(y_labels)
        total = len(y_labels)

        # Verify all labels belong to canonical labels
        for lbl in counts:
            if lbl not in LABEL_TO_INT:
                raise ValueError(f"Unknown label '{lbl}' encountered during MajorityClassifier fit.")

        # Deterministic tie-breaking: highest count, then canonical order
        sorted_classes = sorted(
            CANONICAL_LABELS,
            key=lambda c: (counts.get(c, 0), -LABEL_TO_INT[c]),
            reverse=True,
        )
        self.majority_label = sorted_classes[0]
        self.majority_index = LABEL_TO_INT[self.majority_label]

        self.class_counts = {c: counts.get(c, 0) for c in CANONICAL_LABELS}
        self.class_priors = {c: counts.get(c, 0) / total for c in CANONICAL_LABELS}
        self.prior_distribution = np.array(
            [self.class_priors[c] for c in CANONICAL_LABELS], dtype=float
        )
        self.is_fitted = True
        return self

    def predict(self, X: list[str] | np.ndarray) -> np.ndarray:
        """Predicts the constant majority label for all inputs."""
        if not self.is_fitted or self.majority_label is None:
            raise ValueError("MajorityClassifier must be fitted before predict.")
        n_samples = len(X)
        return np.array([self.majority_label] * n_samples)

    def predict_proba(self, X: list[str] | np.ndarray) -> np.ndarray:
        """Returns empirical training prior probabilities for all inputs."""
        if not self.is_fitted or self.prior_distribution is None:
            raise ValueError("MajorityClassifier must be fitted before predict_proba.")
        n_samples = len(X)
        return np.tile(self.prior_distribution, (n_samples, 1))

    def get_params(self) -> dict[str, Any]:
        """Returns model parameters and training distribution metadata."""
        return {
            "model_type": "MajorityClassifier",
            "majority_label": self.majority_label,
            "majority_index": self.majority_index,
            "class_counts": self.class_counts,
            "class_priors": self.class_priors,
            "is_fitted": self.is_fitted,
        }
