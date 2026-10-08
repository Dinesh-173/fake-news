"""Unit tests verifying strict train/val/test feature isolation and leakage prevention."""

from backend.app.models.baselines.tfidf_logreg import TfidfLogisticRegression
from backend.app.models.baselines.tfidf_svm import TfidfLinearSVM


def test_vectorizer_fits_strictly_on_train():
    """
    Verify that out-of-vocabulary terms unique to validation/test sets
    are strictly NOT present in the fitted TF-IDF vocabulary.
    """
    train_texts = [
        "President announced major infrastructure funding program in capital.",
        "Governor vetoed legislative spending proposal yesterday.",
    ]
    train_labels = ["LIKELY_TRUE", "LIKELY_FALSE"]

    val_texts = [
        "Unprecedented astronomical supernova discovery telescope observations.",
    ]

    model = TfidfLogisticRegression(ngram_range=(1, 1), min_df=1)
    model.fit(train_texts, train_labels)

    # Transforming val_texts must succeed without modifying vocabulary
    X_val = model.vectorizer.transform(val_texts)
    assert X_val.shape[0] == 1
    assert X_val.nnz == 0  # None of the val words are in the training vocabulary

    vocab = model.vectorizer.vocabulary_
    assert "infrastructure" in vocab
    assert "funding" in vocab

    # The words 'supernova', 'telescope', 'astronomical' from val set MUST NOT be in vocab
    assert "supernova" not in vocab
    assert "telescope" not in vocab
    assert "astronomical" not in vocab


def test_svm_vectorizer_fits_strictly_on_train():
    """Verify SVM vectorizer also isolates training vocabulary from evaluation sets."""
    train_texts = [
        "Taxes decreased nationwide following the economic reform.",
        "A controversial rumor circulated regarding public health officials.",
    ]
    train_labels = ["LIKELY_TRUE", "LIKELY_FALSE"]

    model = TfidfLinearSVM(ngram_range=(1, 1), min_df=1)
    model.fit(train_texts, train_labels)

    vocab = model.vectorizer.vocabulary_
    assert "taxes" in vocab
    assert "economic" in vocab
    assert "unseen_evaluation_token" not in vocab
