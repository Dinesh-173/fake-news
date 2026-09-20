# TruthLens: Explainability Strategy

## 1. Multi-Level Interpretability Framework

Explainability in TruthLens operates at three complementary levels to serve diverse stakeholders (readers, journalists, AI researchers):

```text
Level 1: Factual & Evidence Provenance (Reader & Journalist Level)
  - Extracted claims mapped to original text spans
  - Verified source snippets with clickable canonical URLs
  - Explicit stance indicators (Supports / Contradicts / Contextualizes)

Level 2: Multimodal & Cross-Attention Attribution (Auditor Level)
  - Text-Image consistency score and OCR alignment breakdown
  - Claim salience ranking indicating which assertions drove the verdict
  - Evidence weight breakdown showing which source had highest influence

Level 3: Constrained Natural Language Rationale (Executive Summary)
  - Synthesized natural language explanation strictly bound to verified snippets
  - Highlighting key factual contradictions or corroborations
  - Explicit disclosure of missing information or evidence ambiguity
```

---

## 2. Feature Attribution & Token Salience

For deep text models, TruthLens implements token-level attribution via Integrated Gradients or Attention-weighted saliency:
$$\text{Attribution}_i = (x_i - x_i') \times \int_{0}^{1} \frac{\partial F(x' + \alpha (x - x'))}{\partial x_i} d\alpha$$

This reveals whether the model is focusing on factual entity assertions or spurious stylistic markers (e.g., punctuation marks or emotionally charged adverbs).

---

## 3. Explanation Validation & Faithfulness Check

An explanation is defined as **faithful** if and only if every factual assertion in the generated explanation maps to at least one retrieved evidence snippet.

TruthLens runs an automated NLI faithfulness verification pass on LLM-generated explanations:
$$\text{Faithfulness}(Expl, \mathcal{E}) = \frac{1}{|Expl_{\text{sentences}}|} \sum_{s \in Expl} \mathbb{I}\left( \max_{e \in \mathcal{E}} \text{NLI}(e \implies s) \ge \tau_{\text{entail}} \right)$$

If an explanation contains assertions not entailed by retrieved evidence, it is rejected and regenerated with higher constraint temperature.
