"""Baseline C: TF-IDF Vectorizer + Linear Support Vector Machine (LinearSVC)."""

import warnings
from typing import Any

import numpy as np
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from backend.app.models.baselines.evaluator import (
    CANONICAL_LABELS,
    INT_TO_LABEL,
    LABEL_TO_INT,
)


class TfidfLinearSVM:
    """
    Multiclass Linear SVM baseline utilizing sparse word TF-IDF features.
    Guarantees strict isolation: vectorizer is fitted strictly on training data.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_features: int | None = 25000,
        sublinear_tf: bool = True,
        C: float = 1.0,
        class_weight: str | dict | None = None,
        max_iter: int = 2000,
        random_state: int = 42,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            norm="l2",
            strip_accents="unicode",
        )

        self.classifier = LinearSVC(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            dual="auto",
        )

        self.is_fitted: bool = False
        self.iterations_to_converge: int | None = None

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfLinearSVM":
        """
        Fits TF-IDF vectorizer and LinearSVC strictly on the provided training set.
        Never pass validation or test data to fit.
        """
        if len(texts) != len(labels):
            raise ValueError(
                f"Mismatch in input lengths: {len(texts)} texts vs {len(labels)} labels."
            )
        if len(texts) == 0:
            raise ValueError("Cannot fit on empty dataset.")

        # Fit vectorizer strictly on training texts
        X_train = self.vectorizer.fit_transform(texts)

        # Encode labels deterministically
        y_train = np.array([LABEL_TO_INT[lbl] for lbl in labels], dtype=int)

        # Fit classifier with convergence monitoring
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            self.classifier.fit(X_train, y_train)

            conv_warns = [w for w in caught_warnings if issubclass(w.category, ConvergenceWarning)]
            if conv_warns:
                warnings.warn(
                    f"LinearSVC failed to converge within {self.max_iter} iterations.",
                    ConvergenceWarning,
                    stacklevel=2,
                )

        self.iterations_to_converge = int(self.classifier.n_iter_)
        self.is_fitted = True
        return self

    def predict(self, texts: list[str]) -> list[str]:
        """Transforms input texts using pre-fitted vectorizer and predicts canonical labels."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predict.")
        X = self.vectorizer.transform(texts)
        pred_indices = self.classifier.predict(X)
        return [INT_TO_LABEL[int(idx)] for idx in pred_indices]

    def decision_function(self, texts: list[str]) -> np.ndarray:
        """Computes distance of samples to the separating hyperplanes."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before decision_function.")
        X = self.vectorizer.transform(texts)
        return self.classifier.decision_function(X)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Computes normalized pseudo-probabilities using softmax over the SVM decision margins."""
        raw_scores = self.decision_function(texts)
        if raw_scores.ndim == 1:
            raw_scores = np.column_stack([-raw_scores, raw_scores])

        # Compute softmax across classes
        raw_proba = softmax(raw_scores, axis=1)

        # Ensure probabilities align with CANONICAL_LABELS ordering
        aligned_proba = np.zeros((len(texts), len(CANONICAL_LABELS)), dtype=float)
        for class_col_idx, class_val in enumerate(self.classifier.classes_):
            target_idx = int(class_val)
            aligned_proba[:, target_idx] = raw_proba[:, class_col_idx]

        return aligned_proba

    def get_feature_count(self) -> int:
        """Returns the vocabulary feature dimension."""
        if not self.is_fitted:
            return 0
        return len(self.vectorizer.vocabulary_)

    def get_params(self) -> dict[str, Any]:
        """Returns model hyperparameters and configuration state."""
        return {
            "model_type": "TfidfLinearSVM",
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "C": self.C,
            "class_weight": self.class_weight,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "vocabulary_size": self.get_feature_count(),
            "iterations_to_converge": self.iterations_to_converge,
            "is_fitted": self.is_fitted,
        }
