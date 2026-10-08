"""Classical machine learning baselines package for TruthLens."""

from backend.app.models.baselines.evaluator import (
    CANONICAL_LABELS,
    INT_TO_LABEL,
    LABEL_TO_INT,
    BaselineEvaluator,
)
from backend.app.models.baselines.majority import MajorityClassifier
from backend.app.models.baselines.serialization import ModelSerializer
from backend.app.models.baselines.tfidf_logreg import TfidfLogisticRegression
from backend.app.models.baselines.tfidf_svm import TfidfLinearSVM

__all__ = [
    "CANONICAL_LABELS",
    "LABEL_TO_INT",
    "INT_TO_LABEL",
    "BaselineEvaluator",
    "MajorityClassifier",
    "TfidfLogisticRegression",
    "TfidfLinearSVM",
    "ModelSerializer",
]
