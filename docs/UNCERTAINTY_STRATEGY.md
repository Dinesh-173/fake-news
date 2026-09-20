# TruthLens: Uncertainty Quantification & Calibration Strategy

## 1. The Critical Distinction: Confidence vs. Sufficiency

Conventional classifiers equate the maximum softmax probability ($\max_k p_k$) with factual confidence. In open-world news verification, this conflates two distinct forms of confidence:
1. **Prediction Confidence ($\hat{C}_{\text{pred}} \in [0, 1]$)**: How strongly the internal weights and features align with a specific class hypothesis.
2. **Evidence Sufficiency ($S_{\text{evid}} \in [0, 1]$)**: How much relevant, verifiable external reporting exists to substantiate or refute the claim.

A system can be 99% confident that an article *resembles* fake news stylistically while simultaneously possessing 0% external evidence. TruthLens makes this distinction transparent in its API and UI:

```text
┌─────────────────────────────────────────────────────────┐
│ VERIFICATION ASSESSMENT: INSUFFICIENT EVIDENCE          │
│                                                         │
│ Model Feature Probability: 0.88 (Likely False)          │
│ Evidence Sufficiency:      0.15 (Critically Low)        │
│ Decision: Degraded to INSUFFICIENT EVIDENCE             │
│ Reason: No authoritative external sources corroborating │
│         or refuting the claim could be located.         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Temperature Scaling & Calibration

Uncalibrated deep neural networks typically suffer from overconfidence. TruthLens applies post-hoc temperature scaling on a held-out validation split.

Given unnormalized logit vector $\mathbf{z}$:
$$\hat{p}_k(T) = \frac{\exp(z_k / T)}{\sum_{j=1}^K \exp(z_j / T)}$$

Where temperature scalar $T > 0$ is learned by minimizing negative log-likelihood (NLL) on the validation set:
$$T^* = \arg\min_T -\sum_{i=1}^{N_{\text{val}}} \log \hat{p}_{y_i}^{(i)}(T)$$

### Calibration Evaluation Metric
Expected Calibration Error (ECE) is computed with $M=10$ equally spaced confidence bins:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

---

## 3. Epistemic Uncertainty via Monte Carlo Dropout

For deep Transformer backbones, epistemic uncertainty (model parameter uncertainty) is estimated via $S$ stochastic forward passes with dropout enabled during inference:

$$\mu_k = \frac{1}{S} \sum_{s=1}^S \hat{p}_k^{(s)}, \quad \text{Var}_k = \frac{1}{S} \sum_{s=1}^S \left( \hat{p}_k^{(s)} - \mu_k \right)^2$$

High predictive variance signals that the model is extrapolating outside its training distribution.
