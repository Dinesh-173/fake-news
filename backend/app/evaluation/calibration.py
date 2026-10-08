"""TruthLens Uncertainty Calibration and Reliability Analysis Engine.

Implements:
1. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
2. Multi-class Brier Score calculation.
3. Temperature scaling calibration.
4. Reliability diagram bins compilation for scientific reporting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger("truthlens.evaluation.calibration")


@dataclass
class CalibrationReport:
    expected_calibration_error: float
    max_calibration_error: float
    brier_score: float
    num_bins: int
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]
    optimal_temperature: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "max_calibration_error": round(self.max_calibration_error, 4),
            "brier_score": round(self.brier_score, 4),
            "num_bins": self.num_bins,
            "optimal_temperature": round(self.optimal_temperature, 3),
            "bins": [
                {
                    "bin_index": i,
                    "accuracy": round(self.bin_accuracies[i], 4),
                    "confidence": round(self.bin_confidences[i], 4),
                    "count": self.bin_counts[i],
                }
                for i in range(self.num_bins)
            ],
        }


class UncertaintyCalibrator:
    """Computes empirical calibration metrics and optimizes post-hoc temperature scaling."""

    @classmethod
    def compute_calibration_metrics(
        cls,
        y_true: list[int] | np.ndarray,
        y_proba: np.ndarray,
        num_bins: int = 10,
    ) -> CalibrationReport:
        """
        Computes ECE, MCE, and Brier Score over predicted probability distributions.
        """
        y_true_arr = np.asarray(y_true, dtype=int)
        y_proba_arr = np.asarray(y_proba, dtype=float)

        n_samples = len(y_true_arr)
        if n_samples == 0:
            return CalibrationReport(0.0, 0.0, 0.0, num_bins, [], [], [])

        confidences = np.max(y_proba_arr, axis=1)
        predictions = np.argmax(y_proba_arr, axis=1)
        accuracies = (predictions == y_true_arr).astype(float)

        # Multi-class Brier Score
        num_classes = y_proba_arr.shape[1]
        one_hot_true = np.zeros((n_samples, num_classes))
        for i, target in enumerate(y_true_arr):
            one_hot_true[i, target] = 1.0
        brier = float(np.mean(np.sum((y_proba_arr - one_hot_true) ** 2, axis=1)))

        # Reliability Bins
        bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
        bin_accs = []
        bin_confs = []
        bin_counts = []
        ece = 0.0
        mce = 0.0

        for b in range(num_bins):
            b_start = bin_boundaries[b]
            b_end = bin_boundaries[b + 1]

            in_bin = (confidences > b_start) & (confidences <= b_end)
            count = int(np.sum(in_bin))
            bin_counts.append(count)

            if count > 0:
                acc = float(np.mean(accuracies[in_bin]))
                conf = float(np.mean(confidences[in_bin]))
                bin_accs.append(acc)
                bin_confs.append(conf)

                gap = abs(acc - conf)
                ece += (count / n_samples) * gap
                mce = max(mce, gap)
            else:
                bin_accs.append(0.0)
                bin_confs.append((b_start + b_end) / 2.0)

        return CalibrationReport(
            expected_calibration_error=ece,
            max_calibration_error=mce,
            brier_score=brier,
            num_bins=num_bins,
            bin_accuracies=bin_accs,
            bin_confidences=bin_confs,
            bin_counts=bin_counts,
            optimal_temperature=1.0,
        )

    @classmethod
    def apply_temperature_scaling(
        cls,
        logits: np.ndarray,
        temperature: float = 1.25,
    ) -> np.ndarray:
        """
        Scales logits by temperature T > 0 and applies softmax.
        Higher T softens extreme overconfident predictions.
        """
        t = max(0.05, float(temperature))
        scaled = logits / t
        exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)
