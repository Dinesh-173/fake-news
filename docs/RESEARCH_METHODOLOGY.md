# TruthLens: Research Methodology

## 1. Central Research Question

> **Can external evidence retrieval, cross-modal consistency signals, and task-specific deep models produce measurably more reliable, calibrated, and explainable news verification than text-only classifiers or standalone ungrounded Large Language Models?**

This question is framed as an empirical inquiry subject to rigorous falsification rather than an assumed premise.

---

## 2. Research Hypotheses

- **$H_1$ (Evidence Superiority)**: Multimodal models augmented with dynamically retrieved, reranked external evidence achieve statistically significant improvements in Macro-F1 over text-only and closed-book baseline models on ambiguous and evolving news claims.
- **$H_2$ (Cross-Modal Anomaly Detection)**: Explicit modeling of image-text semantic divergence reduces False Positive rates on out-of-context image reuse ("cheap fakes") compared to unimodal text baselines.
- **$H_3$ (Uncertainty Calibration)**: Explicit decoupling of model confidence from evidence sufficiency reduces overconfident misclassifications on out-of-distribution or novel news events.
- **$H_4$ (Explanation Groundedness)**: Evidence-constrained structured prompt templates significantly reduce hallucination rates in verification rationales compared to open-ended LLM chain-of-thought prompting.

---

## 3. Experimental Controls & Protocol

### 3.1 Controlled Split Strategy
To eliminate data contamination:
1. **Deterministic Random Seeds**: Standardized seed ($s = 42$) across all train/val/test partitions.
2. **Entity & Event Disjoint Splitting**: Ensure articles discussing identical specific events or identical source domains do not straddle training and test splits.
3. **Temporal Boundary Splitting**: For datasets with verified publication timestamps, partitions are created chronologically:
   - Training: Historical interval $[t_0, t_1]$
   - Validation: Tuning interval $(t_1, t_2]$
   - Testing: Prospective evaluation interval $(t_2, t_3]$
   This evaluates model robustness under real-world temporal distribution shifts.

### 3.2 Evaluation Metrics
- **Classification Performance**:
  - Accuracy
  - Precision, Recall, Macro-F1, Weighted-F1 (critical due to class imbalance)
  - Multiclass Area Under the ROC Curve (ROC-AUC) via One-vs-Rest (OvR)
  - Precision-Recall AUC (PR-AUC)
- **Calibration & Reliability**:
  - Expected Calibration Error (ECE) across $M = 10$ bins:
    $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
  - Maximum Calibration Error (MCE)
  - Reliability Diagrams plotting predicted confidence against empirical accuracy
- **System Efficiency**:
  - Average inference latency per article (milliseconds)
  - Evidence retrieval latency (breakdown: search, fetch, rerank)
  - Peak memory footprint (RAM and GPU VRAM if available)

### 3.3 Statistical Significance Testing
All baseline comparisons will report 95% confidence intervals derived from $B = 1,000$ bootstrap resamples or paired two-tailed $t$-tests / Wilcoxon signed-rank tests across 5-fold cross-validation folds.
