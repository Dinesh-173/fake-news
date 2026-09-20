# TruthLens: Reproducibility Protocol

## 1. Deterministic Execution Standards

To guarantee identical results when re-running experiments across different systems or dates:

1. **Global Seed Enforcement**: A single centralized function `set_seed(seed: int = 42)` sets seeds across:
   - Python standard `random`
   - NumPy (`np.random.seed`)
   - PyTorch (`torch.manual_seed`, `torch.cuda.manual_seed_all`)
   - PyTorch deterministic algorithms (`torch.use_deterministic_algorithms(True)`)
2. **Environment & Hardware Telemetry**: Every experiment records:
   - Python version and platform architecture
   - Installed package hashes via `pip freeze`
   - Host CPU model, RAM, and GPU model/driver (if CUDA available)
   - Git commit SHA of the exact codebase state during execution

---

## 2. Artifact Directory Standards

All artifacts are persisted in structured, versioned directories:

```text
experiments/
├── registry.json                    # Master manifest of all conducted runs
└── runs/
    └── 2026-09-20_EXP-01_tfidf/
        ├── config.yaml              # Hyperparameters, random seed, dataset
        ├── metrics.json             # Test split evaluation results
        ├── confusion_matrix.png     # Visual confusion matrix plot
        ├── model.joblib             # Serialized model checkpoint
        └── console.log              # Stdout/stderr log of the training run
```

Under no circumstances should prior experiment results be overwritten or altered post-hoc.
