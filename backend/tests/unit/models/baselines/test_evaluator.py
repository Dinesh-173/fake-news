"""Unit tests for BaselineEvaluator multiclass metrics and error analysis."""

import numpy as np
import pytest

from backend.app.models.baselines.evaluator import (
    CANONICAL_LABELS,
    BaselineEvaluator,
)


def test_label_encoding_and_decoding():
    """Verify deterministic bidirectional mapping between strings and integer indices."""
    labels = ["LIKELY_FALSE", "LIKELY_TRUE", "MISLEADING_OR_PARTIALLY_SUPPORTED", "LIKELY_FALSE"]
    encoded = BaselineEvaluator.encode_labels(labels)

    assert isinstance(encoded, np.ndarray)
    assert list(encoded) == [0, 1, 2, 0]

    decoded = BaselineEvaluator.decode_labels(encoded)
    assert decoded == labels


def test_label_encoding_unknown_label():
    """Verify unknown labels raise ValueError."""
    with pytest.raises(ValueError, match="Unknown label"):
        BaselineEvaluator.encode_labels(["INVALID_LABEL"])


def test_evaluator_metrics_computation():
    """Verify multiclass metrics, per-class stats, and confusion matrix dimensions."""
    y_true = ["LIKELY_FALSE", "LIKELY_TRUE", "MISLEADING_OR_PARTIALLY_SUPPORTED", "LIKELY_TRUE"]
    y_pred = ["LIKELY_FALSE", "LIKELY_TRUE", "LIKELY_TRUE", "LIKELY_TRUE"]

    results = BaselineEvaluator.evaluate_predictions(y_true, y_pred)

    overall = results["overall"]
    assert overall["accuracy"] == 0.75
    assert 0.0 <= overall["macro_precision"] <= 1.0
    assert 0.0 <= overall["macro_recall"] <= 1.0
    assert 0.0 <= overall["macro_f1"] <= 1.0

    per_class = results["per_class"]
    for lbl in CANONICAL_LABELS:
        assert lbl in per_class
        assert "precision" in per_class[lbl]
        assert "recall" in per_class[lbl]
        assert "f1_score" in per_class[lbl]
        assert "support" in per_class[lbl]

    assert per_class["LIKELY_FALSE"]["support"] == 1
    assert per_class["LIKELY_TRUE"]["support"] == 2
    assert per_class["MISLEADING_OR_PARTIALLY_SUPPORTED"]["support"] == 1

    # Confusion matrix checks (3x3)
    cm = results["confusion_matrix"]["matrix"]
    assert len(cm) == 3
    assert all(len(row) == 3 for row in cm)
    assert cm[0][0] == 1  # LIKELY_FALSE correctly predicted
    assert cm[1][1] == 2  # LIKELY_TRUE correctly predicted (2 times)
    assert cm[2][1] == 1  # MISLEADING predicted as LIKELY_TRUE (1 time)


def test_evaluator_error_analysis_extraction():
    """Verify deterministic extraction of overconfident, ambiguous, and cell-based errors."""
    records = [
        {"record_id": "r1", "original_id": "1", "text": "Claim 1", "author": "Alice"},
        {"record_id": "r2", "original_id": "2", "text": "Claim 2", "author": "Bob"},
        {"record_id": "r3", "original_id": "3", "text": "Claim 3", "author": "Charlie"},
    ]
    y_true = ["LIKELY_FALSE", "LIKELY_TRUE", "MISLEADING_OR_PARTIALLY_SUPPORTED"]
    y_pred = ["LIKELY_TRUE", "LIKELY_TRUE", "LIKELY_FALSE"]

    # r1: wrong, high confidence (0.90 on LIKELY_TRUE)
    # r2: correct
    # r3: wrong, borderline confidence (0.40 on LIKELY_FALSE, 0.35 on MISLEADING)
    y_proba = np.array([
        [0.05, 0.90, 0.05],  # r1: pred=1 (LIKELY_TRUE, wrong)
        [0.10, 0.80, 0.10],  # r2: pred=1 (LIKELY_TRUE, correct)
        [0.40, 0.25, 0.35],  # r3: pred=0 (LIKELY_FALSE, wrong)
    ])

    errors = BaselineEvaluator.extract_error_analysis(
        records, y_true, y_pred, y_proba, top_k_per_cell=1, top_k_extreme=2
    )

    assert errors["total_errors"] == 2
    assert len(errors["most_overconfident_errors"]) == 2
    assert errors["most_overconfident_errors"][0]["record_id"] == "r1"
    assert errors["most_ambiguous_errors"][0]["record_id"] == "r3"

    # Check confusion cells
    cells = errors["confusion_cell_samples"]
    assert "LIKELY_FALSE -> LIKELY_TRUE" in cells
    assert len(cells["LIKELY_FALSE -> LIKELY_TRUE"]) == 1
    assert cells["LIKELY_FALSE -> LIKELY_TRUE"][0]["record_id"] == "r1"
