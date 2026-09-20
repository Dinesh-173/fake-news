# TruthLens: Experiment Protocol & Benchmark Registry

## 1. Standardized Experiment Lifecycle

```text
[Formulate Hypothesis]
         │
         ▼
[Register Experiment Config] (YAML specifying model, dataset version, seed)
         │
         ▼
[Deterministic Training Run] (Logged step-by-step metrics)
         │
         ▼
[Evaluation on Held-out Splits] (Standard test set, cross-dataset, temporal)
         │
         ▼
[Ablation Analysis] (Leave-one-out feature evaluation)
         │
         ▼
[Error Analysis & Artifact Archival] (Confusion matrix, failure logs, checkpoint)
```

---

## 2. Planned Benchmark Experiments

| Exp ID | Model Architecture | Training Dataset | Evaluation Split | Primary Objective |
| :--- | :--- | :--- | :--- | :--- |
| `EXP-01` | TF-IDF + Logistic Regression | LIAR v1.0 Train | LIAR v1.0 Test | Establish linear lexical baseline |
| `EXP-02` | TF-IDF + LinearSVC | LIAR v1.0 Train | LIAR v1.0 Test | Establish margin-based baseline |
| `EXP-03` | RoBERTa-base | LIAR v1.0 Train | LIAR v1.0 Test | Evaluate deep contextual representation |
| `EXP-04` | RoBERTa + Evidence Stance | LIAR v1.0 Train | LIAR v1.0 Test | Quantify external evidence gain ($H_1$) |
| `EXP-05` | Multimodal (Text + ViT) | FakeNewsNet Train | FakeNewsNet Test | Evaluate multimodal baseline |
| `EXP-06` | TruthLens Full Pipeline | FakeNewsNet Train | FakeNewsNet Test | Full system benchmark |
| `EXP-07` | Cross-Dataset Evaluation | LIAR v1.0 Train | FakeNewsNet Test | Measure out-of-domain transfer robustness |
| `EXP-08` | Temporal Split Evaluation | PHEME / Timeline Train | Prospective Test | Quantify degradation over time |

*Note*: No metric or result may be entered into experiment records until the corresponding training script is executed and evaluated against the dataset split.
