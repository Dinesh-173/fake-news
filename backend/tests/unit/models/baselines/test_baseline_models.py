"""Unit tests for baseline classifiers: Majority, TF-IDF Logistic Regression, and TF-IDF Linear SVM."""

import numpy as np
import pytest

from backend.app.models.baselines.evaluator import CANONICAL_LABELS
from backend.app.models.baselines.majority import MajorityClassifier
from backend.app.models.baselines.tfidf_logreg import TfidfLogisticRegression
from backend.app.models.baselines.tfidf_svm import TfidfLinearSVM


@pytest.fixture
def sample_corpus():
    texts = [
        "The economy grew by three percent in the third quarter.",
        "Unemployment dropped significantly across all regions.",
        "Aliens have established secret bases beneath major cities.",
        "The federal government completely banned sunlight yesterday.",
        "New tax incentives will partially reduce some business costs.",
        "Healthcare spending rose slightly according to the latest report.",
    ]
    labels = [
        "LIKELY_TRUE",
        "LIKELY_TRUE",
        "LIKELY_FALSE",
        "LIKELY_FALSE",
        "MISLEADING_OR_PARTIALLY_SUPPORTED",
        "LIKELY_TRUE",
    ]
    return texts, labels


def test_majority_classifier(sample_corpus):
    """Verify MajorityClassifier identifies majority class and outputs valid predictions/priors."""
    texts, labels = sample_corpus
    clf = MajorityClassifier()

    # Before fit raises ValueError
    with pytest.raises(ValueError, match="must be fitted"):
        clf.predict(["Sample text"])

    clf.fit(texts, labels)

    # LIKELY_TRUE appears 3 times, others appear 2 and 1
    assert clf.majority_label == "LIKELY_TRUE"
    assert clf.class_counts["LIKELY_TRUE"] == 3
    assert clf.class_priors["LIKELY_TRUE"] == 3 / 6

    preds = clf.predict(["Random query 1", "Random query 2"])
    assert list(preds) == ["LIKELY_TRUE", "LIKELY_TRUE"]

    probas = clf.predict_proba(["Random query"])
    assert probas.shape == (1, 3)
    assert np.isclose(np.sum(probas), 1.0)


def test_tfidf_logistic_regression(sample_corpus):
    """Verify TfidfLogisticRegression fits on train and predicts valid labels and probabilities."""
    texts, labels = sample_corpus
    model = TfidfLogisticRegression(ngram_range=(1, 1), min_df=1, C=1.0, random_state=42)

    with pytest.raises(ValueError, match="must be fitted"):
        model.predict(["Unseen text"])

    model.fit(texts, labels)
    assert model.is_fitted
    assert model.get_feature_count() > 0

    test_queries = [
        "The economy and market growth",
        "Aliens and secret bases ban",
    ]
    preds = model.predict(test_queries)
    assert len(preds) == 2
    assert all(p in CANONICAL_LABELS for p in preds)

    probas = model.predict_proba(test_queries)
    assert probas.shape == (2, 3)
    assert np.allclose(np.sum(probas, axis=1), 1.0)


def test_tfidf_linear_svm(sample_corpus):
    """Verify TfidfLinearSVM fits on train and predicts valid labels and decision scores."""
    texts, labels = sample_corpus
    model = TfidfLinearSVM(ngram_range=(1, 1), min_df=1, C=1.0, random_state=42)

    with pytest.raises(ValueError, match="must be fitted"):
        model.predict(["Unseen text"])

    model.fit(texts, labels)
    assert model.is_fitted
    assert model.get_feature_count() > 0

    test_queries = ["Economy report", "Secret alien base"]
    preds = model.predict(test_queries)
    assert len(preds) == 2
    assert all(p in CANONICAL_LABELS for p in preds)

    probas = model.predict_proba(test_queries)
    assert probas.shape == (2, 3)
    assert np.allclose(np.sum(probas, axis=1), 1.0)
