"""Unit tests verifying deterministic reproducibility across identical configurations."""

from backend.app.models.baselines.evaluator import BaselineEvaluator
from backend.app.models.baselines.tfidf_logreg import TfidfLogisticRegression
from backend.app.models.baselines.tfidf_svm import TfidfLinearSVM


def test_reproducibility_logreg():
    """Verify running TfidfLogisticRegression twice with seed 42 produces identical predictions and metrics."""
    texts = [
        "Unemployment dropped in several states according to recent data.",
        "A secret cabal controls international fuel prices completely.",
        "State revenues rose by four percent following the tax adjustment.",
        "Parody article claims gravity has been repealed by Congress.",
    ]
    labels = [
        "LIKELY_TRUE",
        "LIKELY_FALSE",
        "MISLEADING_OR_PARTIALLY_SUPPORTED",
        "LIKELY_FALSE",
    ]

    m1 = TfidfLogisticRegression(random_state=42)
    m1.fit(texts, labels)
    preds1 = m1.predict(texts)
    probas1 = m1.predict_proba(texts)
    eval1 = BaselineEvaluator.evaluate_predictions(labels, preds1, probas1)

    m2 = TfidfLogisticRegression(random_state=42)
    m2.fit(texts, labels)
    preds2 = m2.predict(texts)
    probas2 = m2.predict_proba(texts)
    eval2 = BaselineEvaluator.evaluate_predictions(labels, preds2, probas2)

    assert preds1 == preds2
    assert probas1.tolist() == probas2.tolist()
    assert eval1 == eval2


def test_reproducibility_svm():
    """Verify running TfidfLinearSVM twice with seed 42 produces identical predictions."""
    texts = [
        "Unemployment dropped in several states according to recent data.",
        "A secret cabal controls international fuel prices completely.",
        "State revenues rose by four percent following the tax adjustment.",
        "Parody article claims gravity has been repealed by Congress.",
    ]
    labels = [
        "LIKELY_TRUE",
        "LIKELY_FALSE",
        "MISLEADING_OR_PARTIALLY_SUPPORTED",
        "LIKELY_FALSE",
    ]

    m1 = TfidfLinearSVM(random_state=42)
    m1.fit(texts, labels)
    preds1 = m1.predict(texts)

    m2 = TfidfLinearSVM(random_state=42)
    m2.fit(texts, labels)
    preds2 = m2.predict(texts)

    assert preds1 == preds2
