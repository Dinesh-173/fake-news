"""Encapsulated Transformer Classifier for TruthLens contextual text baselines."""

import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_linear_schedule_with_warmup,
)

from backend.app.models.baselines.evaluator import (
    CANONICAL_LABELS,
    BaselineEvaluator,
)

PINNED_MODELS: dict[str, dict[str, str]] = {
    "bert-base-uncased": {
        "repo_id": "google-bert/bert-base-uncased",
        "revision": "86b5e0934494bd15c9632b12f734a8a67f723594",
        "tokenizer_class": "BertTokenizer",
        "model_class": "BertForSequenceClassification",
    },
    "roberta-base": {
        "repo_id": "FacebookAI/roberta-base",
        "revision": "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
        "tokenizer_class": "RobertaTokenizer",
        "model_class": "RobertaForSequenceClassification",
    },
}


def calculate_balanced_class_weights(
    train_labels: list[int] | list[str] | np.ndarray,
) -> torch.Tensor:
    """
    Computes balanced class weights strictly from training data according to:
        weight_c = N / (K * N_c)
    where:
        N = total training records
        K = 3 (number of canonical classes)
        N_c = count of training records in class c

    Args:
        train_labels: List of integer or string labels from training partition only.

    Returns:
        torch.FloatTensor of shape (3,) with exact weights in order [0, 1, 2].
    """
    from collections import Counter

    if len(train_labels) == 0:
        raise ValueError("Cannot calculate class weights from empty training labels.")

    if isinstance(train_labels[0], str):
        int_labels = BaselineEvaluator.encode_labels(list(train_labels))
    else:
        int_labels = np.asarray(train_labels, dtype=int)

    n_total = len(int_labels)
    k_classes = 3
    counts = Counter(int_labels)

    weights: list[float] = []
    for c in range(k_classes):
        n_c = counts.get(c, 0)
        if n_c == 0:
            raise ValueError(f"Training partition has 0 samples for class index {c} ('{CANONICAL_LABELS[c]}').")
        w_c = float(n_total / (k_classes * n_c))
        weights.append(w_c)

    return torch.tensor(weights, dtype=torch.float32)


class TransformerTextClassifier:
    """
    Modular wrapper for sequence classification transformers (BERT/RoBERTa).

    Guarantees:
    - Immutable revision pinning
    - Deterministic seeding
    - Explicit gradient clipping
    - Native PyTorch training/eval loops without external dependencies
    - Rigorous predicted probabilities extraction without calling softmax 'confidence'
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        revision: str | None = None,
        num_labels: int = 3,
        class_weights: torch.Tensor | list[float] | None = None,
        max_length: int = 128,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.10,
        max_grad_norm: float = 1.0,
        seed: int = 42,
        device: str | torch.device | None = None,
        local_files_only: bool = True,
    ) -> None:
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.max_grad_norm = max_grad_norm
        self.seed = seed

        # Resolve pinned configuration
        if model_name in PINNED_MODELS:
            self.repo_id = PINNED_MODELS[model_name]["repo_id"]
            self.revision = revision or PINNED_MODELS[model_name]["revision"]
            self.expected_tokenizer_class = PINNED_MODELS[model_name]["tokenizer_class"]
            self.expected_model_class = PINNED_MODELS[model_name]["model_class"]
        else:
            self.repo_id = model_name
            self.revision = revision or "main"
            self.expected_tokenizer_class = "AutoTokenizer"
            self.expected_model_class = "AutoModelForSequenceClassification"

        # Device assignment
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Class weights
        if class_weights is not None:
            if isinstance(class_weights, list):
                self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            else:
                self.class_weights = class_weights.to(self.device)
        else:
            self.class_weights = None

        self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)

        # Set deterministic seeds
        self.set_seed(self.seed)

        # Instantiate tokenizer and model
        self.tokenizer: PreTrainedTokenizerBase = self._load_tokenizer(local_files_only)
        self.model: PreTrainedModel = self._load_model(local_files_only)
        self.model.to(self.device)

        # Optimization components (built upon training start)
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any = None

    @staticmethod
    def set_seed(seed: int) -> None:
        """Enforces deterministic pseudo-random seeds across Python, NumPy, and PyTorch."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def _load_tokenizer(self, local_files_only: bool) -> PreTrainedTokenizerBase:
        """Loads tokenizer pinned to immutable revision."""
        try:
            return AutoTokenizer.from_pretrained(
                self.repo_id,
                revision=self.revision,
                local_files_only=local_files_only,
            )
        except Exception:
            # Fallback if local_files_only was requested but assets not yet staged
            return AutoTokenizer.from_pretrained(
                self.repo_id,
                revision=self.revision,
                local_files_only=False,
            )

    def _load_model(self, local_files_only: bool) -> PreTrainedModel:
        """Loads sequence classification head pinned to immutable revision."""
        try:
            return AutoModelForSequenceClassification.from_pretrained(
                self.repo_id,
                num_labels=self.num_labels,
                revision=self.revision,
                local_files_only=local_files_only,
            )
        except Exception:
            return AutoModelForSequenceClassification.from_pretrained(
                self.repo_id,
                num_labels=self.num_labels,
                revision=self.revision,
                local_files_only=False,
            )

    def build_optimizer_and_scheduler(self, total_training_steps: int) -> None:
        """Constructs AdamW optimizer with decoupled weight decay and linear warmup schedule."""
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        num_warmup_steps = int(total_training_steps * self.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
        )

    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Computes loss using the configured CrossEntropyLoss (standard or class-weighted)."""
        return self.loss_fn(logits, labels)

    def training_step(self, batch: dict[str, torch.Tensor]) -> float:
        """Executes one single forward-backward-step optimization sequence with gradient clipping."""
        if self.optimizer is None or self.scheduler is None:
            raise RuntimeError("Optimizer and scheduler not initialized. Call build_optimizer_and_scheduler first.")

        self.model.train()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        loss = self.compute_loss(outputs.logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        return float(loss.item())

    def evaluate(self, dataloader: DataLoader) -> dict[str, Any]:
        """
        Runs full read-only inference across validation dataloader.

        Returns:
            Dictionary containing 'eval_loss', 'overall' metrics, 'per_class' metrics,
            and confusion matrix data.
        """
        self.model.eval()
        total_eval_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_probas: list[list[float]] = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.compute_loss(outputs.logits, labels)
                total_eval_loss += float(loss.item()) * len(labels)

                logits = outputs.logits
                probas = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = np.argmax(probas, axis=-1)

                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
                all_probas.extend(probas.tolist())

        n_samples = len(all_labels)
        avg_eval_loss = total_eval_loss / n_samples if n_samples > 0 else 0.0

        # Compute empirical metrics via BaselineEvaluator
        eval_metrics = BaselineEvaluator.evaluate_predictions(
            y_true_raw=all_labels,
            y_pred_raw=all_preds,
            y_proba=np.array(all_probas),
        )
        eval_metrics["eval_loss"] = round(avg_eval_loss, 4)
        eval_metrics["total_samples"] = n_samples

        return eval_metrics

    def predict_proba(
        self, dataloader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Extracts predicted probabilities (softmax distributions) and predictions.

        Note: Probabilities represent raw uncalibrated predicted probabilities.
        """
        self.model.eval()
        all_preds: list[int] = []
        all_probas: list[list[float]] = []
        all_record_ids: list[str] = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probas = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                preds = np.argmax(probas, axis=-1)

                all_preds.extend(preds.tolist())
                all_probas.extend(probas.tolist())
                if "record_id" in batch:
                    all_record_ids.extend(batch["record_id"])

        return np.array(all_preds), np.array(all_probas), all_record_ids

    def get_parameter_counts(self) -> dict[str, int]:
        """Returns total and trainable parameter counts."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total_parameters": total, "trainable_parameters": trainable}

    def get_metadata(self) -> dict[str, Any]:
        """Compiles complete model configuration and telemetry metadata."""
        import transformers

        param_counts = self.get_parameter_counts()
        return {
            "model_name": self.model_name,
            "repository_id": self.repo_id,
            "revision_sha": self.revision,
            "tokenizer_class": type(self.tokenizer).__name__,
            "model_architecture": type(self.model).__name__,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
            "seed": self.seed,
            "device": str(self.device),
            "class_weights": [round(float(w), 5) for w in self.class_weights] if self.class_weights is not None else None,
            "total_parameters": param_counts["total_parameters"],
            "trainable_parameters": param_counts["trainable_parameters"],
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
        }
