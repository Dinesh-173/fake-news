"""Unit tests for TruthLens calibration, uncertainty, and robustness evaluation engines."""

import numpy as np
import pytest

from backend.app.evaluation.calibration import UncertaintyCalibrator
from backend.app.evaluation.robustness import RobustnessEvaluator, TextPerturber


def test_uncertainty_calibration_metrics():
    # Synthetic ground truth: 3 classes
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    # Well-calibrated simulated probabilities
    y_proba = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.85, 0.1, 0.05],
    ])

    report = UncertaintyCalibrator.compute_calibration_metrics(y_true, y_proba, num_bins=5)
    assert report.num_bins == 5
    assert 0.0 <= report.expected_calibration_error <= 1.0
    assert 0.0 <= report.max_calibration_error <= 1.0
    assert 0.0 <= report.brier_score <= 2.0
    assert len(report.bin_accuracies) == 5
    assert len(report.bin_counts) == 5

    d = report.to_dict()
    assert "expected_calibration_error" in d
    assert "brier_score" in d
    assert len(d["bins"]) == 5


def test_temperature_scaling():
    logits = np.array([
        [2.0, 1.0, 0.0],
        [0.0, 3.0, 1.0],
    ])
    # T = 1.0 (standard softmax)
    p1 = UncertaintyCalibrator.apply_temperature_scaling(logits, temperature=1.0)
    assert np.allclose(np.sum(p1, axis=1), 1.0)
    assert p1[0, 0] > p1[0, 1] > p1[0, 2]

    # High T = softer distributions (closer to uniform)
    p_high = UncertaintyCalibrator.apply_temperature_scaling(logits, temperature=10.0)
    assert p_high[0, 0] < p1[0, 0]
    assert np.allclose(np.sum(p_high, axis=1), 1.0)


def test_text_perturber_primitives():
    original = "The unemployment rate dropped significantly across the state."

    # Typos
    typo_text = TextPerturber.apply_typos(original, rate=0.2, seed=42)
    assert len(typo_text) == len(original)
    assert typo_text != original

    # Casing
    cased_text = TextPerturber.apply_casing_noise(original, seed=42)
    assert cased_text != original

    # Paraphrase prefix
    prefixed = TextPerturber.apply_paraphrase_prefix(original, seed=42)
    assert len(prefixed) > len(original)

    # Length
    short_v = TextPerturber.apply_extreme_length(original, mode="short")
    long_v = TextPerturber.apply_extreme_length(original, mode="long")
    assert len(short_v.split()) <= 4
    assert len(long_v) > len(original) * 2


def test_robustness_evaluator():
    sample_claims = [
        "Health care costs surged by fifteen percent in rural clinics.",
        "Tax revenues exceeded official government forecasts this fiscal quarter.",
        "Independent audits revealed unaccounted expenditures in highway maintenance.",
    ]

    # Deterministic dummy predictor
    def mock_predictor(text: str) -> dict:
        # If words like "costs" or "taxes" present, predict MISLEADING
        if "tax" in text.lower():
            return {"label": "LIKELY_TRUE", "confidence": 0.82}
        return {"label": "MISLEADING_OR_PARTIALLY_SUPPORTED", "confidence": 0.65}

    evaluator = RobustnessEvaluator(mock_predictor)
    report = evaluator.run_suite(sample_claims)

    assert report.num_samples_evaluated == 3
    assert len(report.summaries) == 5
    for summary in report.summaries:
        assert 0.0 <= summary.invariance_rate <= 1.0
        assert summary.error_rate == 0.0

    report_dict = report.to_dict()
    assert report_dict["num_samples_evaluated"] == 3
    assert len(report_dict["summaries"]) == 5
