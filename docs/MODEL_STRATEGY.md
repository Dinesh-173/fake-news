# TruthLens: Model Strategy & Baseline Progression

## 1. Incremental Baseline Architecture

To ensure scientific rigor, TruthLens adopts an incremental evaluation hierarchy. Every advanced architecture must justify its computational and operational complexity against established baselines.

```text
Complexity Level 1: Classical Baselines
  - Baseline A: TF-IDF + Logistic Regression
  - Baseline B: TF-IDF + Linear Support Vector Classifier (LinearSVC)

Complexity Level 2: Deep Unimodal Text Baselines
  - Baseline C: Pretrained Transformer (RoBERTa-base / DeBERTa-v3-small)
  - Baseline D: Long-context Transformer (Longformer / BigBird for full articles)

Complexity Level 3: Multimodal Unimodal/Dual Baselines
  - Baseline E: Unimodal Vision (ViT-B/16 image-only classifier)
  - Baseline F: Early Fusion (Concatenated RoBERTa text + ViT image features)

Complexity Level 4: Evidence-Aware Architectures
  - Baseline G: Text + Retrieved External Evidence (NLI Stance Cross-Attention)
  - Target: TruthLens Full Architecture (Multimodal + Evidence Stance + Uncertainty Calibration)
```

---

## 2. Model Specifications & Training Protocols

### 2.1 Classical Baselines (A & B)
- **Features**: Sublinear TF-IDF n-grams (unigrams and bigrams, $n \in [1, 2]$), vocabulary capped at 20,000 features, English stop-word removal.
- **Hyperparameter Optimization**: 5-fold cross-validated grid search over regularization strength $C \in [0.01, 0.1, 1.0, 10.0]$ and penalty formulations (L1, L2).
- **Purpose**: Establishes minimum viable lexical signal threshold.

### 2.2 Deep Text Transformers (C & D)
- **Backbone**: `roberta-base` or `microsoft/deberta-v3-base`.
- **Max Sequence Length**: 512 tokens (truncated/pooled with headline preservation).
- **Optimization**: AdamW ($\beta_1=0.9, \beta_2=0.999$, $\epsilon=1\times 10^{-8}$), weight decay 0.01, linear learning rate warmup for 10% of total steps, learning rate $\eta = 2\times 10^{-5}$.
- **Loss Function**: Cross-Entropy with optional focal loss or class-weight adjustment for imbalanced datasets.

### 2.3 Vision Encoder (E)
- **Backbone**: `google/vit-base-patch16-224` or ResNet-50.
- **Preprocessing**: Resize to $224 \times 224$, standard ImageNet normalization ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

---

## 3. Training Reproducibility & Artifact Storage

Every model training run produces an immutable checkpoint bundle under `models/<experiment_id>/` containing:
1. `model_weights.pt` or `safetensors`: Serialized model parameters.
2. `config.json`: Model architecture parameters, random seeds, and hyperparameters.
3. `preprocessor.json` / `tokenizer/`: Exact tokenizer configurations and vocabulary files.
4. `metrics.json`: Evaluated test metrics (Accuracy, F1, ECE, Confusion Matrix).
5. `training_log.jsonl`: Step-by-step train/validation loss and learning rate decay records.
