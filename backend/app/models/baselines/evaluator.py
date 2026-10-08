"""Multiclass baseline evaluation, metric computation, and deterministic error analysis."""

from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

CANONICAL_LABELS = [
    "LIKELY_FALSE",
    "LIKELY_TRUE",
    "MISLEADING_OR_PARTIALLY_SUPPORTED",
]

LABEL_TO_INT = {lbl: i for i, lbl in enumerate(CANONICAL_LABELS)}
INT_TO_LABEL = dict(enumerate(CANONICAL_LABELS))


class BaselineEvaluator:
    """Computes comprehensive multiclass performance metrics and structured error analysis."""

    @classmethod
    def encode_labels(cls, labels: list[str]) -> np.ndarray:
        """Deterministically encodes canonical string labels into integer indices."""
        for lbl in labels:
            if lbl not in LABEL_TO_INT:
                raise ValueError(
                    f"Unknown label '{lbl}'. Expected one of: {list(LABEL_TO_INT.keys())}"
                )
        return np.array([LABEL_TO_INT[lbl] for lbl in labels], dtype=int)

    @classmethod
    def decode_labels(cls, indices: np.ndarray | list[int]) -> list[str]:
        """Decodes integer indices back to canonical string labels."""
        return [INT_TO_LABEL[int(idx)] for idx in indices]

    @classmethod
    def evaluate_predictions(
        cls,
        y_true_raw: list[str] | np.ndarray,
        y_pred_raw: list[str] | np.ndarray,
        y_proba: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Computes overall, per-class, and confusion matrix metrics.
        Guarantees deterministic class ordering: LIKELY_FALSE (0), LIKELY_TRUE (1), MISLEADING_OR_PARTIALLY_SUPPORTED (2).
        """
        if isinstance(y_true_raw[0], str):
            y_true = cls.encode_labels(list(y_true_raw))
        else:
            y_true = np.asarray(y_true_raw, dtype=int)

        if isinstance(y_pred_raw[0], str):
            y_pred = cls.encode_labels(list(y_pred_raw))
        else:
            y_pred = np.asarray(y_pred_raw, dtype=int)

        total_samples = len(y_true)
        if total_samples == 0:
            raise ValueError("Cannot evaluate empty prediction arrays.")

        acc = float(accuracy_score(y_true, y_pred))
        macro_p = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        macro_r = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        weighted_p = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        weighted_r = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        # Per-class metrics
        class_precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
        class_recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
        class_f1s = f1_score(y_true, y_pred, average=None, zero_division=0)
        class_supports = Counter(y_true)

        per_class: dict[str, dict[str, Any]] = {}
        for idx, lbl in enumerate(CANONICAL_LABELS):
            per_class[lbl] = {
                "precision": float(class_precisions[idx]),
                "recall": float(class_recalls[idx]),
                "f1_score": float(class_f1s[idx]),
                "support": int(class_supports.get(idx, 0)),
            }

        # Confusion matrix (rows: true, cols: pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        cm_list = cm.tolist()

        # Normalized confusion matrix (percentage per ground-truth row)
        cm_normalized = []
        for row in cm:
            row_sum = sum(row)
            if row_sum > 0:
                cm_normalized.append([round(val / row_sum * 100, 2) for val in row])
            else:
                cm_normalized.append([0.0] * len(row))

        results: dict[str, Any] = {
            "total_samples": total_samples,
            "overall": {
                "accuracy": round(acc, 4),
                "macro_precision": round(macro_p, 4),
                "macro_recall": round(macro_r, 4),
                "macro_f1": round(macro_f1, 4),
                "weighted_precision": round(weighted_p, 4),
                "weighted_recall": round(weighted_r, 4),
                "weighted_f1": round(weighted_f1, 4),
            },
            "per_class": per_class,
            "confusion_matrix": {
                "classes": CANONICAL_LABELS,
                "matrix": cm_list,
                "matrix_normalized_percent": cm_normalized,
            },
        }

        return results

    @classmethod
    def extract_error_analysis(
        cls,
        records: list[dict[str, Any]],
        y_true_raw: list[str],
        y_pred_raw: list[str],
        y_proba: np.ndarray,
        top_k_per_cell: int = 2,
        top_k_extreme: int = 5,
    ) -> dict[str, Any]:
        """
        Extracts representative errors deterministically:
        1. Top-K most overconfident errors (high probability on wrong class)
        2. Top-K least confident errors (ambiguous boundary predictions)
        3. Systematic cell-by-cell errors across all off-diagonal confusion matrix pairs
        """
        y_true_idx = cls.encode_labels(y_true_raw)
        y_pred_idx = cls.encode_labels(y_pred_raw)

        errors: list[dict[str, Any]] = []
        for i in range(len(records)):
            true_lbl = y_true_raw[i]
            pred_lbl = y_pred_raw[i]
            if true_lbl != pred_lbl:
                prob_pred = float(y_proba[i, y_pred_idx[i]])
                prob_true = float(y_proba[i, y_true_idx[i]])
                errors.append(
                    {
                        "index": i,
                        "record_id": records[i].get("record_id", ""),
                        "original_id": records[i].get("original_id", ""),
                        "statement": records[i].get("text", ""),
                        "speaker": records[i].get("author", "") or records[i].get("metadata", {}).get("speaker", ""),
                        "ground_truth": true_lbl,
                        "predicted": pred_lbl,
                        "predicted_probability": round(prob_pred, 4),
                        "confidence": round(prob_pred, 4),  # Alias for predicted_probability (uncalibrated)
                        "true_class_probability": round(prob_true, 4),
                        "probabilities": {
                            lbl: round(float(y_proba[i, idx]), 4)
                            for idx, lbl in enumerate(CANONICAL_LABELS)
                        },
                    }
                )

        # Sort by confidence descending (most overconfident errors)
        errors_sorted_desc = sorted(errors, key=lambda x: x["confidence"], reverse=True)
        overconfident_errors = errors_sorted_desc[:top_k_extreme]

        # Sort by confidence ascending (lowest confidence / ambiguous errors)
        errors_sorted_asc = sorted(errors, key=lambda x: x["confidence"])
        ambiguous_errors = errors_sorted_asc[:top_k_extreme]

        # Off-diagonal confusion cells
        cell_errors: dict[str, list[dict[str, Any]]] = {}
        for true_lbl in CANONICAL_LABELS:
            for pred_lbl in CANONICAL_LABELS:
                if true_lbl != pred_lbl:
                    pair_key = f"{true_lbl} -> {pred_lbl}"
                    matching = [
                        e for e in errors
                        if e["ground_truth"] == true_lbl and e["predicted"] == pred_lbl
                    ]
                    matching_sorted = sorted(matching, key=lambda x: x["confidence"], reverse=True)
                    cell_errors[pair_key] = matching_sorted[:top_k_per_cell]

        return {
            "total_errors": len(errors),
            "error_rate": round(len(errors) / max(len(records), 1), 4),
            "most_overconfident_errors": overconfident_errors,
            "most_ambiguous_errors": ambiguous_errors,
            "confusion_cell_samples": cell_errors,
        }
